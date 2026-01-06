from pathlib import Path
from lxml import etree
import sqlite3
import re


# Paths


INPUT_DIR = Path(r"S:\src\pamphlets\eebo_all\eebo_phase1\P4_XML_TCP")
OUT_DIR = Path(r"S:\src\pamphlets\out")
PLAIN_DIR = OUT_DIR / "plain"
DB_PATH = OUT_DIR / "metadata.sqlite"

PLAIN_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_early_modern(text: str) -> str:
    text = text.lower()

    text = text.replace("ſ", "s")
    text = re.sub(r'-\s+', '', text)
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


# Main processing


def process_file(xml_path, conn):
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        print(f"[ERROR] XML parse failed: {xml_path.name}: {e}")
        return False

    # ---------- Metadata ----------
    doc_id = extract_first(tree, "//HEADER//IDNO[@TYPE='DLPS']/text()")
    if not doc_id:
        return False

    title = extract_first(tree, "//HEADER//TITLESTMT/TITLE/text()")
    author = extract_first(tree, "//HEADER//TITLESTMT/AUTHOR/text()")

    date_raw = extract_first(tree, "//HEADER//SOURCEDESC//DATE/text()")
    pub_year = extract_pub_year(date_raw)

    publisher = extract_first(tree, "//HEADER//SOURCEDESC//PUBLISHER/text()")
    pub_place = extract_first(tree, "//HEADER//SOURCEDESC//PUBPLACE/text()")

    conn.execute("""
        INSERT OR REPLACE INTO documents (
        doc_id, title, author, pub_year, publisher,
        pub_place, source_date_raw
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        doc_id,
        title,
        author,
        pub_year,
        publisher,
        pub_place,
        date_raw
    ))

    # ---------- Body text ----------
    body_text_nodes = tree.xpath("//EEBO/BODY//text()")
    if not body_text_nodes:
        return False

    raw_text = " ".join(t.strip() for t in body_text_nodes if t.strip())
    normalized = normalize_early_modern(raw_text)

    if len(normalized) < 100:
        return False

    out_path = PLAIN_DIR / f"{doc_id}.txt"
    out_path.write_text(normalized, encoding="utf-8")

    return True


# Entry point


def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    xml_files = list(INPUT_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    processed = 0

    for i, xml_file in enumerate(xml_files, 1):
        if process_file(xml_file, conn):
            processed += 1

        if i % 500 == 0:
            conn.commit()
            print(f"Processed {i} files ({processed} usable texts)")

    conn.commit()
    conn.close()

    print(f"Done. {processed} texts written to {PLAIN_DIR}")
    print(f"Metadata database: {DB_PATH}")


if __name__ == "__main__":
    main()
