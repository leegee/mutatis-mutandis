#!/usr/bin/env python
from lxml import etree
import sqlite3
import re
import unicodedata
import sys

import eebo_config as config
import eebo_db
import eebo_ocr_fixes
import eebo_slice

# Ensure output directories exist
try:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.PLAIN_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Cannot create output directories: {e}")
    sys.exit(1)


print(f"[INFO] SQLite DB will be created at: {config.DB_PATH}")


import re
import unicodedata

def normalize_early_modern(text: str) -> str:
    # Lowercase
    text = text.lower()

    # normalize all apostrophes to simple ASCII '
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)

    # Long s
    text = text.replace("ſ", "s")

    # Unicode normalization to remove accents, etc
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Join hyphenated line breaks
    text = re.sub(r'-\s*', ' ', text)

    # Initial u/v, j/i
    text = re.sub(r'\bv(?=[aeiou])', 'u', text)
    text = re.sub(r'\bj(?=[aeiou])', 'i', text)

    # Remove OCR artefacts inside words
    text = re.sub(r'(?<=\w)[^\w\s](?=\w)', '', text)

    # Remove remaining non-letter characters but keep spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def extract_first(tree, xpath):
    result = tree.xpath(xpath)
    if result:
        return result[0].strip()
    return None


def extract_pub_year(date_str):
    if not date_str:
        return None
    match = re.search(r'\b(1[5-7][0-9]{2})\b', date_str)
    return int(match.group(1)) if match else None


def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            pub_year INTEGER,
            publisher TEXT,
            pub_place TEXT,
            source_date_raw TEXT
        )
    """)
    conn.commit()


def process_file(xml_path, conn):
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        print(f"[ERROR] XML parse failed: {xml_path.name}: {e}")
        return False

    # Metadata
    doc_id = extract_first(tree, "//HEADER//IDNO[@TYPE='DLPS']/text()")
    if not doc_id:
        print(f"[WARN] No DLPS ID found in {xml_path.name}")
        return False

    title = extract_first(tree, "//HEADER//TITLESTMT/TITLE/text()")
    author = extract_first(tree, "//HEADER//TITLESTMT/AUTHOR/text()")
    date_raw = extract_first(tree, "//HEADER//SOURCEDESC//DATE/text()")
    pub_year = extract_pub_year(date_raw)
    publisher = extract_first(tree, "//HEADER//SOURCEDESC//PUBLISHER/text()")
    pub_place = extract_first(tree, "//HEADER//SOURCEDESC//PUBPLACE/text()")

    try:
        conn.execute("""
            INSERT OR REPLACE INTO documents
            (doc_id, title, author, pub_year, publisher,
            pub_place, source_date_raw)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            title,
            author,
            pub_year,
            publisher,
            pub_place,
            date_raw
        ))
    except Exception as e:
        print(f"[ERROR] Failed to insert metadata for {doc_id}: {e}")
        return False

    # Body text
    eebo_elem = tree.find(".//EEBO")
    if eebo_elem is None:
        print(f"[WARN] No <EEBO> element for {doc_id}")
        return False

    body = eebo_elem.find(".//TEXT/BODY")
    if body is None:
        print(f"[WARN] No <BODY> inside <EEBO> for {doc_id}")
        return False

    body_text_nodes = list(body.itertext())

    raw_text = " ".join(t.strip() for t in body_text_nodes if t.strip())
    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)

    if len(normalized) < 100:
        print(f"[WARN] Skipping very short text for {doc_id}")
        return False

    out_path = config.PLAIN_DIR / f"{doc_id}.txt"
    try:
        out_path.write_text(normalized, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write text file for {doc_id}: {e}")
        return False

    return True


def main():
    init_db(eebo_db.dbh)

    xml_files = list(config.INPUT_DIR.glob("*.xml"))
    print(f"[INFO] Found {len(xml_files)} XML files")

    processed = 0

    for i, xml_file in enumerate(xml_files, 1):
        if process_file(xml_file, eebo_db.dbh):
            processed += 1

        if i % 500 == 0:
            eebo_db.dbh.commit()
            print(f"[INFO] Processed {i} files ({processed} usable texts)")

    eebo_db.dbh.commit()
    eebo_db.dbh.close()

    print(f"[DONE] {processed} texts written to {config.PLAIN_DIR}")
    print(f"[DONE] Metadata database: {config.DB_PATH}")

    eebo_slice.make_slices()

if __name__ == "__main__":
    main()
