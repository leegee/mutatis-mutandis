#!/usr/bin/env python
"""
eebo_parse_tei.py - Multi-process streaming EEBO TEI XML ingestion pipeline
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import argparse
import os
import csv
import io
import re
import sys
import tempfile
import unicodedata
import traceback
from concurrent.futures import ProcessPoolExecutor

from psycopg import sql
import xml.etree.ElementTree as etree
import langdetect
import lib.eebo_config as config
import lib.eebo_db as eebo_db
import lib.eebo_ocr_fixes as eebo_ocr_fixes
from lib.eebo_logging import logger
from lib.set_lang import set_document_languages

MAX_DOCS: Optional[int] = None
INGEST_ALL = True

LOG_EVERY_N_DOCS = 100

ALLOWED_PUNCT = r"\.\,\;\:\!\?\'\"\-\(\)"


def normalize_early_modern(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(\w)[’‘ʼ′´](\w)", r"\1'\2", text)
    text = text.replace("ſ", "s")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    text = re.sub(r"-\s*", " ", text)
    text = re.sub(r"\bv(?=[aeiou])", "u", text)
    text = re.sub(r"\bj(?=[aeiou])", "i", text)
    text = re.sub(r"tv\b", "ty", text)

    text = re.sub(rf"[^{ALLOWED_PUNCT}a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def render_text(node):
    """Render EEBO XML into plain text while preserving editorial GAP markers."""
    parts = []

    if node.text:
        parts.append(node.text)

    for child in node:
        if child.tag.upper() == "GAP":
            extent = child.attrib.get("EXTENT", "")
            m = re.search(r"(\d+)", extent)
            n = int(m.group(1)) if m else 1
            parts.append("_" * n)
        else:
            parts.append(render_text(child))

        if child.tail:
            parts.append(child.tail)

    return "".join(parts)


def extract_year_and_slice(date_raw: str | None):
    if not date_raw:
        return None, None, None

    m = re.search(r"\b(\d{4})\b", date_raw)
    if not m:
        return None, None, None

    pub_year = int(m.group(1))

    for start, end in config.SLICES:
        if start <= pub_year <= end:
            return start, end, pub_year

    return None, None, None


def safe_text(x):
    return x.text.strip() if x is not None and x.text else None


def to_doc_row(meta: dict) -> tuple:
    return (
        meta["doc_id"],
        meta["title"],
        meta["author"],
        meta["pub_year"],
        meta["publisher"],
        meta["pub_place"],
        meta["source_date_raw"],
        meta["token_count"],
        meta.get("filepath"),
        meta.get("lang"),
        # meta["slice_start"],
        # meta["slice_end"],
    )


def process_file(xml_path: Path):
    try:
        tree = etree.parse(str(xml_path))
    except Exception:
        logger.warning(f"Failed to parse {xml_path}")
        return None

    doc_id_elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    if doc_id_elem is None or not doc_id_elem.text:
        return None

    doc_id = doc_id_elem.text.strip()

    lang = tree.findtext(".//PROFILEDESC//LANGUAGE")
    if (lang == "en"):
        lang = "eng"
    elif not lang:
        try:
            detected_lang = langdetect.detect(raw_text[:5000])
        except Exception:
            detected_lang = None
        lang = detected_lang

    title_elem = tree.find(".//HEADER//TITLESTMT/TITLE")
    author_elem = tree.find(".//HEADER//TITLESTMT/AUTHOR")
    pub_elem = tree.find(".//HEADER//SOURCEDESC//PUBLISHER")
    place_elem = tree.find(".//HEADER//SOURCEDESC//PUBPLACE")
    date_elem = tree.find(".//HEADER//SOURCEDESC//DATE")

    date_raw = safe_text(date_elem)
    slice_start, slice_end, pub_year = extract_year_and_slice(date_raw)

    if pub_year is None:
        return None

    body = tree.findall(".//EEBO//TEXT//BODY")
    if not body:
        return None

    raw_text = " ".join(render_text(b) for b in body)

    normalized = normalize_early_modern(
        eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    )

    if len(normalized) < 100:
        return None

    tokens = re.findall(r"\w+|[^\w\s]", normalized)

    meta = {
        "doc_id": doc_id,
        "title": safe_text(title_elem),
        "author": safe_text(author_elem),
        "publisher": safe_text(pub_elem),
        "pub_place": safe_text(place_elem),
        "pub_year": pub_year,
        "source_date_raw": date_raw,
        "token_count": len(tokens),
        "filepath": str(xml_path.relative_to(config.XML_ROOT_DIR)),
        "lang": lang,
    }

    return meta, tokens


def process_file_to_temp(xml_path: Path):
    result = process_file(xml_path)
    if not result:
        return None

    meta, tokens = result

    tmp = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="", suffix=".tsv")
    writer = csv.writer(tmp, delimiter="\t")

    for i, tok in enumerate(tokens):          # Simplified
        writer.writerow([meta["doc_id"], i, tok])

    tmp.close()

    return meta, tmp.name, len(tokens)


def stream_copy(table: str, columns: list[str], rows):
    if not rows:
        return

    stmt = sql.SQL(
        "COPY {table} ({fields}) FROM STDIN WITH (FORMAT text, DELIMITER E'\t', NULL '\\N')"
    ).format(
        table=sql.Identifier(table),
        fields=sql.SQL(', ').join(sql.Identifier(c) for c in columns)
    )

    def encode(row):
        return "\t".join(
            "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
            for v in row
        ) + "\n"

    buf = io.StringIO()
    for r in rows:
        buf.write(encode(r))
    buf.seek(0)

    with eebo_db.get_autocommit_connection() as conn:
        with conn.cursor() as cur:
            with cur.copy(stmt) as copy:
                copy.write(buf.read())


def _worker_ingest(files, batch_docs, batch_tokens, ingest_all):
    doc_batch = []
    token_batch = []
    inserted_doc_ids = set()

    docs_seen = 0

    def log_progress():
        if docs_seen % LOG_EVERY_N_DOCS == 0 and docs_seen > 0:
            logger.info(f"[worker {os.getpid()}] ingested {docs_seen} docs")

    def flush_docs():
        nonlocal doc_batch, inserted_doc_ids
        if not doc_batch:
            return

        rows = doc_batch
        doc_batch = []

        if not ingest_all:
            rows = filter_existing_docs(rows)   # Make sure this function exists

        if not rows:
            return

        inserted_doc_ids.update(r[0] for r in rows)

        stream_copy(
            "documents",
            [
                "doc_id", "title", "author", "pub_year",
                "publisher", "pub_place", "source_date_raw",
                "token_count", "filepath", "lang", # "slice_start", "slice_end"
            ],
            rows,
        )

    def flush_tokens():
        nonlocal token_batch
        if not token_batch:
            return

        rows = [t for t in token_batch if t[0] in inserted_doc_ids]
        token_batch = []

        if rows:
            stream_copy("tokens", ["doc_id", "token_idx", "token"], rows)

    for fp in files:
        try:
            result = process_file_to_temp(fp)
            if not result:
                continue

            meta, token_file, _ = result

            doc_batch.append(to_doc_row(meta))

            with open(token_file, "r", encoding="utf-8") as f:
                for line in f:
                    doc_id, idx, tok = line.rstrip("\n").split("\t")
                    token_batch.append((doc_id, int(idx), tok))

            if len(doc_batch) >= batch_docs:
                flush_docs()

            if len(token_batch) >= batch_tokens:
                flush_tokens()

            docs_seen += 1
            log_progress()

        except Exception:
            logger.error(f"FAILED FILE: {fp}")
            logger.error(traceback.format_exc())

    flush_docs()
    flush_tokens()
    logger.info(f"[worker {os.getpid()}] finished: {docs_seen} docs processed")


def ingest_xml_parallel(xml_dir: Path, max_workers: int = 4, batch_docs: int = 50, batch_tokens: int = 50000):
    xml_files = list(Path(xml_dir).rglob("*.xml"))

    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    chunks = [xml_files[i::max_workers] for i in range(max_workers)]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_worker_ingest, chunk, batch_docs, batch_tokens, INGEST_ALL)
            for chunk in chunks
        ]
        for f in futures:
            f.result()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--justindex", action="store_true")

    args = parser.parse_args()

    global MAX_DOCS, INGEST_ALL
    MAX_DOCS = args.limit
    INGEST_ALL = args.create or False

    with eebo_db.get_connection() as conn:
        if args.create:
            confirm = input("DESTROY DB? type YES: ")
            if confirm != "YES":
                sys.exit(1)
            eebo_db.init_db(conn)

        eebo_db.drop_token_indexes(conn)
        eebo_db.drop_tokens_fk(conn)
        conn.commit()

    if not args.justindex:
        ingest_xml_parallel(
            xml_dir=config.XML_ROOT_DIR,
            max_workers=config.NUM_WORKERS,
            batch_docs=config.BATCH_DOCS,
            batch_tokens=config.BATCH_TOKENS,
        )
        # set_document_languages() - try to avoid this using lang detection above

    # Wait for other connections to finish
    with eebo_db.get_connection() as conn:
        while True:
            cur = conn.execute("""
                SELECT count(*) FROM pg_stat_activity
                WHERE datname = 'eebo'
                  AND pid <> pg_backend_pid()
                  AND state IN ('active', 'idle in transaction');
            """)
            n = cur.fetchone()[0]
            if n == 0:
                break

    with eebo_db.get_connection() as conn:
        eebo_db.create_tokens_fk(conn)
        eebo_db.create_token_indexes(conn)
        eebo_db.create_tiered_token_indexes(conn)
        eebo_db.refresh_views(conn)

    eebo_db.create_concurrent_indexes()


if __name__ == "__main__":
    main()
