#!/usr/bin/env python
# eebo_parse_tei.py
"""
Multi-process streaming EEBO TEI XML ingestion pipeline

- Multi-process XML parsing
- Progressive streaming COPY into final tables
- Safe re-ingest
- Call with `--limit int` or all documents in the target dir will be processed
- See `eebo_config.py`
"""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm
import argparse
import io
import re
import unicodedata
import tempfile
import csv
import os

import eebo_config as config
import eebo_db
import eebo_ocr_fixes
from eebo_logging import logger

MAX_DOCS: Optional[int] = None

def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"-\s*", " ", text)
    text = re.sub(r"\bv(?=[aeiou])", "u", text)
    text = re.sub(r"\bj(?=[aeiou])", "i", text)
    text = re.sub(r"(?<=\w)[^\w\s](?=\w)", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_year(date_raw: str | None) -> Optional[int]:
    if not date_raw:
        return None
    m = re.search(r"\b(\d{4})\b", date_raw)
    return int(m.group(1)) if m else None

def assign_slice(date_raw: str | None) -> tuple[int | None, int | None]:
    if not date_raw:
        return None, None
    m = re.search(r"\b(\d{4})\b", date_raw)
    if not m:
        return None, None
    year = int(m.group(1))
    for start, end in config.SLICES:
        if start <= year <= end:
            return start, end
    return None, None

def process_file(xml_path: Path) -> Optional[tuple[dict[str, Any], list[tuple[int, str]]]]:
    """Parse a single XML file and return doc metadata + token list."""
    import xml.etree.ElementTree as etree

    try:
        tree = etree.parse(str(xml_path))
    except Exception as exc:
        logger.error(f"XML parse failed: {xml_path.name}: {exc}")
        return None

    doc_id_elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    if doc_id_elem is None or not doc_id_elem.text:
        logger.error(f"XML rejected: {xml_path.name}: missing document ID")
        return None

    doc_id = doc_id_elem.text.strip()

    title_elem = tree.find(".//HEADER//TITLESTMT/TITLE")
    author_elem = tree.find(".//HEADER//TITLESTMT/AUTHOR")
    date_elem = tree.find(".//HEADER//SOURCEDESC//DATE")
    pub_elem = tree.find(".//HEADER//SOURCEDESC//PUBLISHER")
    place_elem = tree.find(".//HEADER//SOURCEDESC//PUBPLACE")

    title = title_elem.text.strip() if title_elem is not None and title_elem.text else None
    author = author_elem.text.strip() if author_elem is not None and author_elem.text else None
    date_raw = date_elem.text.strip() if date_elem is not None and date_elem.text else None
    publisher = pub_elem.text.strip() if pub_elem is not None and pub_elem.text else None
    pub_place = place_elem.text.strip() if place_elem is not None and place_elem.text else None

    slice_start, slice_end = assign_slice(date_raw)

    body_elems = tree.findall(".//EEBO//TEXT//BODY")
    if not body_elems:
        logger.error(f"XML rejected: {xml_path.name} at {xml_path.resolve()}: no body text found")
        return None

    raw_text = " ".join(
        t.strip()
        for body in body_elems
        for t in body.itertext()
        if t.strip()
    )

    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)

    if len(normalized) < 100:
        logger.error(
            f"XML rejected (text too short): {xml_path.name} at {xml_path.resolve()} "
            f"({len(normalized)} chars after normalization)"
        )
        return None

    tokens = normalized.split()
    token_rows = [(i, tok) for i, tok in enumerate(tokens)]

    doc_meta = {
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "publisher": publisher,
        "pub_place": pub_place,
        "source_date_raw": date_raw,
        "slice_start": slice_start,
        "slice_end": slice_end,
    }

    return doc_meta, token_rows

def process_file_to_temp(xml_path: Path) -> Optional[tuple[dict[str, Any], str]]:
    """Parse a single XML file, write tokens to a temporary file, return metadata + temp file path."""
    result = process_file(xml_path)
    if not result:
        return None

    doc_meta, token_tuples = result

    # Create a temp file for tokens
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".tsv")
    writer = csv.writer(temp_file, delimiter="\t")
    for token_idx, token in token_tuples:
        writer.writerow([doc_meta["doc_id"], token_idx, token])
    temp_file.close()  # flush and close so main process can read

    return doc_meta, temp_file.name

def stream_copy(conn, table: str, columns: list[str], rows):
    """Streaming COPY directly into final table."""
    if not rows:
        return

    sql = (
        f"COPY {table} ({', '.join(columns)}) "
        "FROM STDIN WITH (FORMAT text, DELIMITER E'\t', NULL '\\N')"
    )

    buf = io.StringIO()
    for row in rows:
        buf.write(
            "\t".join(
                "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
                for v in row
            ) + "\n"
        )
    buf.seek(0)

    with conn.cursor() as cur:
        with cur.copy(sql) as copy:
            copy.write(buf.read())

def ingest_xml_parallel_safe(max_workers: int = 4, batch_docs: int = 100, batch_tokens: int = 10000) -> None:
    """Main ingestion function using temp files for tokens and batching."""
    xml_files = list(Path(config.INPUT_DIR).glob("*.xml"))
    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    if not xml_files:
        logger.warning("No XML files found for ingestion")
        return

    logger.info(f"Starting ingestion of {len(xml_files)} XML files")

    doc_rows: list[list[Any]] = []

    # Step 1: parse XML in parallel, write tokens to temp files
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file_to_temp, p): p for p in xml_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing XML"):
            xml_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error(f"XML worker failed for {xml_path.name}: {exc}")
                continue

            if result:
                results.append(result)

    logger.info(f"Parsing complete. {len(results)} documents ready for DB insert")

    # Step 2: write metadata + tokens to DB in batches
    with eebo_db.get_connection() as conn:
        with conn.transaction():
            token_batch: list[list[Any]] = []
            for doc_meta, token_file in tqdm(results, desc="Writing to DB"):
                doc_id = doc_meta["doc_id"]

                # Document metadata
                doc_rows.append([
                    doc_id,
                    doc_meta["title"],
                    doc_meta["author"],
                    extract_year(doc_meta["source_date_raw"]),
                    doc_meta["publisher"],
                    doc_meta["pub_place"],
                    doc_meta["source_date_raw"],
                    sum(1 for _ in open(token_file, "r")),  # token count
                    doc_meta["slice_start"],
                    doc_meta["slice_end"],
                ])

                # Flush document metadata if batch is full
                if len(doc_rows) >= batch_docs:
                    logger.info(f"Flushing {len(doc_rows)} document metadata")
                    stream_copy(
                        conn,
                        "documents",
                        [
                            "doc_id", "title", "author", "pub_year",
                            "publisher", "pub_place", "source_date_raw",
                            "token_count", "slice_start", "slice_end"
                        ],
                        doc_rows,
                    )
                    doc_rows.clear()

                # Read tokens and batch for COPY
                with open(token_file, "r") as f:
                    for line in f:
                        token_batch.append(line.strip().split("\t"))
                        if len(token_batch) >= batch_tokens:
                            stream_copy(
                                conn,
                                "tokens",
                                ["doc_id", "token_idx", "token"],
                                token_batch,
                            )
                            token_batch.clear()

                # Remove temp file
                os.remove(token_file)

            # Final flush of remaining doc metadata
            if doc_rows:
                logger.info(f"Final flush of {len(doc_rows)} document metadata")
                stream_copy(
                    conn,
                    "documents",
                    [
                        "doc_id", "title", "author", "pub_year",
                        "publisher", "pub_place", "source_date_raw",
                        "token_count", "slice_start", "slice_end"
                    ],
                    doc_rows,
                )

            # Final flush of remaining token batch
            if token_batch:
                logger.info(f"Final flush of {len(token_batch)} tokens")
                stream_copy(
                    conn,
                    "tokens",
                    ["doc_id", "token_idx", "token"],
                    token_batch,
                )

    logger.info("XML ingestion complete")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max documents to process")
    args = parser.parse_args()

    global MAX_DOCS
    MAX_DOCS = args.limit

    with eebo_db.get_connection() as conn:
        eebo_db.init_db(conn)

    ingest_xml_parallel_safe()

if __name__ == "__main__":
    main()
