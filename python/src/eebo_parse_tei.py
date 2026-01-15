#!/usr/bin/env python
"""
EEBO ingestion + processing pipeline.

Notes:
- XML ingestion is batch-streamed via COPY into staging tables.
- Sentence building and canonicalisation are batch-streamed, not fully streaming.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import eebo_config as config
import eebo_db
import eebo_ocr_fixes
from eebo_logging import logger

# ---------------------------------------------------------------------------
# Globals / model setup
# ---------------------------------------------------------------------------

MAX_DOCS: Optional[int] = None  # --limit testing only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_PATH,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    config.MODEL_PATH,
    local_files_only=True,
).to(DEVICE)

model.eval()

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalize_early_modern(text: str) -> str:
    """Normalize Early Modern English text for tokenisation."""
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
    match = re.search(r"\b(\d{4})\b", date_raw)
    return int(match.group(1)) if match else None


def assign_slice(year: Optional[int]) -> tuple[Optional[int], Optional[int]]:
    if year is None:
        return None, None
    for start, end in config.SLICES:
        if start <= year <= end:
            return start, end
    return None, None


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------


def process_file(
    xml_path: Path,
) -> Optional[tuple[dict[str, Any], list[tuple[int, str]]]]:
    """Parse XML file and return metadata + token list."""
    import xml.etree.ElementTree as etree

    try:
        tree = etree.parse(str(xml_path))
    except Exception as exc:  # noqa: BLE001
        logger.error("XML parse failed: %s: %s", xml_path.name, exc)
        return None

    doc_id_elem = tree.find(".//HEADER//IDNO[@TYPE='DLPS']")
    doc_id = doc_id_elem.text.strip() if doc_id_elem is not None and doc_id_elem.text else None
    if not doc_id:
        logger.error("XML rejected: %s: missing document ID", xml_path.name)
        return None

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

    year = extract_year(date_raw)
    slice_start, slice_end = assign_slice(year)

    body_elem = tree.find(".//EEBO//TEXT//BODY")
    if body_elem is None:
        logger.error("XML rejected: %s: no body text", xml_path.name)
        return None

    raw_text = " ".join(t.strip() for t in body_elem.itertext() if t.strip())
    fixed_text = eebo_ocr_fixes.apply_ocr_fixes(raw_text)
    normalized = normalize_early_modern(fixed_text)

    if len(normalized) < 100:
        logger.error("XML rejected: %s: text too short", xml_path.name)
        return None

    tokens = normalized.split()
    token_rows = [(i, tok) for i, tok in enumerate(tokens)]

    meta = {
        "doc_id": doc_id,
        "title": title,
        "author": author,
        "pub_year": year,
        "publisher": publisher,
        "pub_place": pub_place,
        "source_date_raw": date_raw,
        "token_count": len(tokens),
        "slice_start": slice_start,
        "slice_end": slice_end,
    }

    return meta, token_rows


# ---------------------------------------------------------------------------
# COPY helpers
# ---------------------------------------------------------------------------


def stream_copy(conn, table: str, columns: list[str], rows) -> None:
    cols = ", ".join(columns)
    sql = (
        f"COPY {table} ({cols}) "
        "FROM STDIN WITH (FORMAT text, NULL '\\N', DELIMITER E'\\t')"
    )

    with conn.cursor() as cur, cur.copy(sql) as copy:
        for row in rows:
            line = "\t".join(
                "\\N" if v is None else str(v).replace("\t", " ").replace("\n", " ")
                for v in row
            )
            copy.write(line + "\n")


def flush_staging_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                doc_id, title, author, pub_year, publisher,
                pub_place, source_date_raw, token_count,
                slice_start, slice_end
            )
            SELECT DISTINCT ON (doc_id)
                doc_id, title, author, pub_year, publisher,
                pub_place, source_date_raw, token_count,
                slice_start, slice_end
            FROM documents_stage
            ON CONFLICT (doc_id) DO NOTHING
            """
        )

        cur.execute(
            """
            INSERT INTO tokens (doc_id, token_idx, token)
            SELECT doc_id, token_idx, token
            FROM tokens_stage
            ON CONFLICT DO NOTHING
            """
        )

        cur.execute("TRUNCATE documents_stage")
        cur.execute("TRUNCATE tokens_stage")


# ---------------------------------------------------------------------------
# Phase 1 — XML ingestion
# ---------------------------------------------------------------------------


def ingest_xml_parallel(max_workers: int = 4, batch_docs: int = 100) -> None:
    xml_files = list(Path(config.INPUT_DIR).glob("*.xml"))
    if MAX_DOCS:
        xml_files = xml_files[:MAX_DOCS]

    if not xml_files:
        logger.warning("No XML files found for ingestion.")
        return

    logger.info("Starting streaming ingestion of %d XML files", len(xml_files))

    doc_rows: list[list[Any]] = []
    token_rows: list[list[Any]] = []
    docs_in_batch = 0

    with eebo_db.get_connection() as conn, conn.transaction():
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_file, p): p for p in xml_files}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Ingesting XML"):
                result = future.result()
                if not result:
                    continue

                meta, tokens = result
                doc_rows.append([
                    meta["doc_id"],
                    meta["title"],
                    meta["author"],
                    meta["pub_year"],
                    meta["publisher"],
                    meta["pub_place"],
                    meta["source_date_raw"],
                    meta["token_count"],
                    meta["slice_start"],
                    meta["slice_end"],
                ])

                for idx, tok in tokens:
                    token_rows.append([meta["doc_id"], idx, tok])

                docs_in_batch += 1

                if docs_in_batch >= batch_docs:
                    stream_copy(
                        conn,
                        "documents_stage",
                        [
                            "doc_id", "title", "author", "pub_year",
                            "publisher", "pub_place", "source_date_raw",
                            "token_count", "slice_start", "slice_end",
                        ],
                        doc_rows,
                    )
                    stream_copy(
                        conn,
                        "tokens_stage",
                        ["doc_id", "token_idx", "token"],
                        token_rows,
                    )
                    flush_staging_tables(conn)

                    doc_rows.clear()
                    token_rows.clear()
                    docs_in_batch = 0

        if doc_rows:
            stream_copy(
                conn,
                "documents_stage",
                [
                    "doc_id", "title", "author", "pub_year",
                    "publisher", "pub_place", "source_date_raw",
                    "token_count", "slice_start", "slice_end",
                ],
                doc_rows,
            )
            stream_copy(
                conn,
                "tokens_stage",
                ["doc_id", "token_idx", "token"],
                token_rows,
            )
            flush_staging_tables(conn)

    logger.info("XML ingestion complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices={"ingest", "sentences", "canonicalize", "all"},
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    global MAX_DOCS
    MAX_DOCS = args.limit

    eebo_db.drop_token_indexes()
    ingest_xml_parallel()
    eebo_db.create_token_indexes()

    logger.info("Ingestion completed successfully.")


if __name__ == "__main__":
    main()
