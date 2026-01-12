#!/usr/bin/env python
from lxml import etree
from pathlib import Path
from io import StringIO
import re
import unicodedata

import eebo_config as config
import eebo_db
import eebo_ocr_fixes
import eebo_logging

logger = eebo_logging.setup_logging("eebo.ingest")

# Set to `None` to process all XMLs; set to integer for testing a limited number
MAX_DOCS = 100


def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r'-\s*', ' ', text)
    text = re.sub(r'\bv(?=[aeiou])', 'u', text)
    text = re.sub(r'\bj(?=[aeiou])', 'i', text)
    text = re.sub(r'(?<=\w)[^\w\s](?=\w)', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_first(tree: etree._ElementTree, xpath: str) -> str | None:
    result = tree.xpath(xpath)
    return result[0].strip() if result else None


def extract_pub_year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    m = re.search(r'\b(1[5-7][0-9]{2})\b', date_str)
    return int(m.group(1)) if m else None


def assign_slice(pub_year: int | None) -> tuple[int | None, int | None]:
    if pub_year is None:
        return None, None
    for start, end in config.SLICES:
        if start <= pub_year <= end:
            return start, end
    return None, None


def insert_tokens(doc_id: str, normalized_text: str, conn) -> int:
    tokens = normalized_text.split()
    buf = StringIO()
    for idx, tok in enumerate(tokens):
        buf.write(f"{doc_id}\t{idx}\t{tok}\n")
    buf.seek(0)

    with conn.cursor() as cur:
        # Delete existing tokens for idempotence
        cur.execute("DELETE FROM tokens WHERE doc_id = %s", (doc_id,))
        with cur.copy("COPY tokens (doc_id, token_idx, token) FROM STDIN") as copy:
            copy.write(buf.read())

    return len(tokens)


def process_file(xml_path: Path, conn) -> bool:
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:
        logger.error(f"XML parse failed: {xml_path.name}: {e}")
        return False

    doc_id = extract_first(tree, "//HEADER//IDNO[@TYPE='DLPS']/text()")
    if not doc_id:
        logger.error(f"XML parse failed: {xml_path.name}: no document ID at //HEADER//IDNO[@TYPE='DLPS']/text()")
        return False

    title = extract_first(tree, "//HEADER//TITLESTMT/TITLE/text()")
    author = extract_first(tree, "//HEADER//TITLESTMT/AUTHOR/text()")
    date_raw = extract_first(tree, "//HEADER//SOURCEDESC//DATE/text()")
    pub_year = extract_pub_year(date_raw)
    publisher = extract_first(tree, "//HEADER//SOURCEDESC//PUBLISHER/text()")
    pub_place = extract_first(tree, "//HEADER//SOURCEDESC//PUBPLACE/text()")

    slice_start, slice_end = assign_slice(pub_year)

    # Upsert document metadata
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                doc_id, title, author, pub_year,
                publisher, pub_place, source_date_raw,
                slice_start, slice_end
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                pub_year = EXCLUDED.pub_year,
                publisher = EXCLUDED.publisher,
                pub_place = EXCLUDED.pub_place,
                source_date_raw = EXCLUDED.source_date_raw,
                slice_start = EXCLUDED.slice_start,
                slice_end = EXCLUDED.slice_end
            """,
            (doc_id, title, author, pub_year, publisher, pub_place, date_raw, slice_start, slice_end)
        )

    eebo_elem = tree.find(".//EEBO")
    if eebo_elem is None:
        logger.error(f"XML parse failed: {xml_path.name}: no .//EEBO node")
        return False

    body = eebo_elem.find(".//TEXT/BODY")
    if body is None:
        logger.error(f"XML rejected: {xml_path.name}: no .//TEXT/BODY")
        return False

    raw_text = " ".join(t.strip() for t in body.itertext() if t.strip())
    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)

    if len(normalized) < 100:
        logger.error(f"XML rejected: {xml_path.name}: .//TEXT/BODY length is less than 100")
        return False

    token_count = insert_tokens(doc_id, normalized, conn)

    with conn.cursor() as cur:
        cur.execute("UPDATE documents SET token_count = %s WHERE doc_id = %s", (token_count, doc_id))

    return True


def main():
    eebo_db.init_db()
    eebo_db.drop_token_indexes()

    xml_files = list(config.INPUT_DIR.glob("*.xml"))
    if MAX_DOCS is not None:
        xml_files = xml_files[:MAX_DOCS]

    logger.info(f"Found {len(xml_files)} XML files")

    processed = 0
    for i, xml_file in enumerate(xml_files, 1):
        if process_file(xml_file, eebo_db.dbh):
            processed += 1
        if i % 500 == 0:
            eebo_db.dbh.commit()
            logger.info(f"Processed {i} files ({processed} usable)")

    eebo_db.dbh.commit()
    eebo_db.create_token_indexes()
    eebo_db.dbh.close()
    logger.info(f"Ingested {processed} documents")


if __name__ == "__main__":
    main()
