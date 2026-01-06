#!/usr/bin/env python3
from lxml import etree
import sqlite3
import re
import sys

import eebo_config

# Ensure output directories exist
try:
    eebo_config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    eebo_config.PLAIN_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Cannot create output directories: {e}")
    sys.exit(1)


# Check SQLite3 can create DB
try:
    conn_test = sqlite3.connect(eebo_config.DB_PATH)
    conn_test.execute("CREATE TABLE IF NOT EXISTS _test (id INTEGER);")
    conn_test.execute("DROP TABLE _test;")
    conn_test.commit()
    conn_test.close()
except Exception as e:
    print(
        f"[ERROR] Cannot create or write to SQLite DB at {eebo_config.DB_PATH}: {e}"
    )
    sys.exit(1)

print(f"[INFO] SQLite DB will be created at: {eebo_config.DB_PATH}")


def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = text.replace("ſ", "s")           # long s
    text = re.sub(r'-\s+', '', text)        # join hyphenated line breaks
    text = re.sub(r'\bv(?=[aeiou])', 'u', text)
    text = re.sub(r'\bj(?=[aeiou])', 'i', text)
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

    body_text_nodes = list(eebo_elem.itertext())
    if not body_text_nodes:
        print(f"[WARN] <EEBO> has no text for {doc_id}")
        return False

    raw_text = " ".join(t.strip() for t in body_text_nodes if t.strip())
    normalized = normalize_early_modern(raw_text)

    if len(normalized) < 100:
        print(f"[WARN] Skipping very short text for {doc_id}")
        return False

    out_path = eebo_config.PLAIN_DIR / f"{doc_id}.txt"
    try:
        out_path.write_text(normalized, encoding="utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to write text file for {doc_id}: {e}")
        return False

    return True


def main():
    try:
        conn = sqlite3.connect(eebo_config.DB_PATH)
    except Exception as e:
        print(f"[ERROR] Cannot open SQLite DB at {eebo_config.DB_PATH}: {e}")
        sys.exit(1)

    init_db(conn)

    xml_files = list(eebo_config.INPUT_DIR.glob("*.xml"))
    print(f"[INFO] Found {len(xml_files)} XML files")

    processed = 0

    for i, xml_file in enumerate(xml_files, 1):
        if process_file(xml_file, conn):
            processed += 1

        if i % 500 == 0:
            conn.commit()
            print(f"[INFO] Processed {i} files ({processed} usable texts)")
        else:
            print(f"[DEBUG] Skipped file: {xml_file.name}")

    conn.commit()
    conn.close()

    print(f"[DONE] {processed} texts written to {eebo_config.PLAIN_DIR}")
    print(f"[DONE] Metadata database: {eebo_config.DB_PATH}")


if __name__ == "__main__":
    main()
