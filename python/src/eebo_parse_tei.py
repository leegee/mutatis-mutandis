#!/usr/bin/env python
"""
eebo_parse_tei.py - Multi-process streaming EEBO TEI XML ingestion pipeline

- Multi-process XML parsing
- Progressive streaming COPY into final tables
- Safe re-ingest
- Call with `--limit int` or all documents in the target dir will be processed
- See `eebo_config.py`

To do:

- re-introduce `processPoolExecutor`with `hunksize`
- re-introduce `xml`
- re-introduce concurrent db cx

"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional, Iterator
import argparse
import csv
import io
import os
import re
import sys
import tempfile
import unicodedata
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from psycopg import sql
import xml.etree.ElementTree as etree

import lib.eebo_config as config
import lib.eebo_db as eebo_db
import lib.eebo_ocr_fixes as eebo_ocr_fixes
from lib.eebo_logging import logger
from lib.set_lang import set_document_languages

MAX_DOCS: Optional[int] = None


def normalize_early_modern_preserve_punct(text: str) -> str:
    """
    Normalize Early Modern text while preserving punctuation tokens
    like . , ; : ! ? ' " - ( )
    """
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalize print variants
    text = re.sub(r"-\s*", " ", text)
    text = re.sub(r"\bv(?=[aeiou])", "u", text)
    text = re.sub(r"\bj(?=[aeiou])", "i", text)
    text = re.sub(r"tv\b", "ty", text)

    # Preserve punctuation we care about: replace any other non-word/non-space char with space
    allowed_punct = r"\.\,\;\:\!\?\'\"\-\(\)"
    text = re.sub(f"[^{allowed_punct}a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Restore OCR artifacts to obvious intentions
    text = re.sub(r"\bliberti\s+s\b", "liberties", text)
    text = re.sub(r"\bliber\s+ies\b", "liberties", text)
    text = re.sub(r"\blibe\s+ty\b", "liberty", text)
    text = re.sub(r"\baliberty\b", "a liberty", text)
    text = re.sub(r"\bl\s+berty\b", "liberty", text)
    text = re.sub(r"\bberty\b", "liberty", text)
    text = re.sub(r"\blibe\s+y\b", "liberty", text)

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


def filter_existing_docs(doc_batch):
    """Removes from doc_batch docs already in the DB and returns the result"""
    if not doc_batch:
        return []

    doc_ids = [doc[0] for doc in doc_batch]

    with eebo_db.get_autocommit_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id FROM documents WHERE doc_id = ANY(%s)",
                (doc_ids,)
            )
            existing = {row[0] for row in cur.fetchall()}

    # Keep only docs not already in DB
    new_batch = [doc for doc in doc_batch if doc[0] not in existing]

    skipped = len(doc_ids) - len(new_batch)
    if skipped:
        logger.info(f"Skipping {skipped} documents already in DB")

    return new_batch


def process_file(xml_path: Path) -> Optional[tuple[dict[str, Any], list[tuple[int, str]]]]:
    """Parse a single XML file and return doc metadata + token list."""
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

    title = re.sub(r'\W+', '', title_elem.text.strip()) if title_elem is not None and title_elem.text else None
    author = re.sub(r'\W+', '', author_elem.text.strip()) if author_elem is not None and author_elem.text else None
    date_raw = re.sub(r'\W+', '', date_elem.text.strip()) if date_elem is not None and date_elem.text else None
    publisher = re.sub(r'\W+', '', pub_elem.text.strip()) if pub_elem is not None and pub_elem.text else None
    pub_place = re.sub(r'\W+', '', place_elem.text.strip()) if place_elem is not None and place_elem.text else None

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

    # tokens = normalized.split() # Without punctuation
    tokens = re.findall(r"\w+|[^\w\s]", normalized) # with punctuation

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


def process_file_to_temp(xml_path: Path) -> Optional[tuple[dict[str, Any], str, int]]:
    """Parse XML, write tokens to temp file, return metadata + temp file path + token count."""
    result = process_file(xml_path)
    if not result:
        return None

    doc_meta, token_tuples = result

    token_count = 0
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".tsv")
    writer = csv.writer(temp_file, delimiter="\t")
    for token_idx, token in token_tuples:
        writer.writerow([doc_meta["doc_id"], token_idx, token])
        token_count += 1
    temp_file.close()

    return doc_meta, temp_file.name, token_count


def stream_copy(table: str, columns: list[str], rows):
    """Use COPY to bulk-insert rows safely in a fresh autocommit connection."""
    if not rows:
        return

    logger.info(f"Streaming {len(rows)} rows into {table}")

    with eebo_db.get_autocommit_connection() as conn:
        stmt = sql.SQL(
            "COPY {table} ({fields}) FROM STDIN WITH (FORMAT text, DELIMITER E'\t', NULL '\\N')"
        ).format(
            table=sql.Identifier(table),
            fields=sql.SQL(', ').join(sql.Identifier(c) for c in columns)
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
            with cur.copy(stmt) as copy:
                copy.write(buf.read())


def ingest_xml_parallel(
    xml_dir: Path = config.XML_ROOT_DIR,
    max_workers: int = 4,
    batch_docs: int = config.BATCH_DOCS,
    batch_tokens: int = config.BATCH_TOKENS
) -> None:
    """Stream XML parsing + incremental DB ingestion with bounded memory."""

    xml_iter:  Iterator[Path] = Path(xml_dir).rglob("*.xml")
    if MAX_DOCS:
        xml_iter = islice(xml_iter, MAX_DOCS)

    logger.info("Starting streaming ingestion")

    doc_batch: list[list[Any]] = []
    token_batch: list[list[Any]] = []
    inserted_doc_ids: set[str] = set()

    # bound number of in-flight tasks to avoid memory blowup
    max_in_flight = max_workers * 2

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = set()

        def flush_docs():
            nonlocal doc_batch, inserted_doc_ids
            if not doc_batch:
                return

            filtered = filter_existing_docs(doc_batch)
            if filtered:
                new_ids = {doc[0] for doc in filtered}
                inserted_doc_ids.update(new_ids)

                logger.info(f"Flushing {len(filtered)} documents")
                stream_copy(
                    "documents",
                    [
                        "doc_id", "title", "author", "pub_year",
                        "publisher", "pub_place", "source_date_raw",
                        "token_count", "slice_start", "slice_end"
                    ],
                    filtered
                )
            doc_batch.clear()

        def flush_tokens():
            nonlocal token_batch
            if not token_batch:
                return

            filtered = [t for t in token_batch if t[0] in inserted_doc_ids]
            if filtered:
                logger.info(f"Flushing {len(filtered)} tokens")
                stream_copy(
                    "tokens",
                    ["doc_id", "token_idx", "token"],
                    filtered
                )
            token_batch.clear()

        def handle_result(result):
            if not result:
                return

            doc_meta, token_file, token_count = result
            doc_id = doc_meta["doc_id"]

            doc_batch.append([
                doc_id,
                doc_meta["title"],
                doc_meta["author"],
                extract_year(doc_meta["source_date_raw"]),
                doc_meta["publisher"],
                doc_meta["pub_place"],
                doc_meta["source_date_raw"],
                token_count,
                doc_meta["slice_start"],
                doc_meta["slice_end"],
            ])

            with open(token_file, "r") as f:
                for line in f:
                    token_batch.append(line.strip().split("\t"))

            os.remove(token_file)

        # prime the pipeline
        try:
            for _ in range(max_in_flight):
                futures.add(executor.submit(process_file_to_temp, next(xml_iter)))
        except StopIteration:
            pass

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)

            for fut in done:
                try:
                    result = fut.result()
                except Exception as exc:
                    logger.error(f"Worker failed: {exc}")
                    continue

                handle_result(result)

                # refill pipeline
                try:
                    futures.add(executor.submit(process_file_to_temp, next(xml_iter)))
                except StopIteration:
                    pass

            # flush docs first so tokens have valid FK targets
            if len(doc_batch) >= batch_docs:
                flush_docs()

            if len(token_batch) >= batch_tokens:
                flush_tokens()

        # final flush
        flush_docs()
        flush_tokens()

    logger.info("XML ingestion complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EEBO-TCP XML Ingest Script: Ingests EEBO-TCP XML into PostgreSQL with optional DB reset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to process"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Re-create the DB losing all data"
    )
    args = parser.parse_args()

    global MAX_DOCS
    MAX_DOCS = args.limit

    with eebo_db.get_connection() as conn:

        if args.create:
            confirm = input( "WARNING: This will DESTROY the current database and re-create it. Type YES to proceed: " )
            if confirm != "YES":
                logger.info('Aborted by user')
                sys.exit(1)
            logger.info('Initialising DB')
            eebo_db.init_db(conn)
            logger.info('DB initialised')

        eebo_db.drop_token_indexes(conn)
        eebo_db.drop_tokens_fk(conn)
        conn.commit()

    logger.info('DB initialised, ingesting XML')
    logger.info(f"Processing max {MAX_DOCS if MAX_DOCS else 'all'} documents")

    ingest_xml_parallel(
        xml_dir=config.BASE_DIR / "eebo_all",
        max_workers=config.NUM_WORKERS,
        batch_docs=config.BATCH_DOCS,
        batch_tokens=config.BATCH_TOKENS
    )

    set_document_languages()

    logger.info('All ingested, restoring indexes')
    with eebo_db.get_connection() as conn:
        eebo_db.create_tokens_fk(conn)
        eebo_db.create_token_indexes(conn)
        eebo_db.create_tiered_token_indexes(conn)
        conn.commit()

    logger.info('Rebuilding views')
    eebo_db.refresh_views(conn)

    logger.info('Done - all ingested, indexes restored')


if __name__ == "__main__":
    main()
