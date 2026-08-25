#!/usr/bin/env python
"""
lookup_events.py

Read search hits (TSV or JSON from tier1_5_query_lance.py) and enrich
each event_id with:
  - observation metadata from the Parquet event store
  - document metadata from PostgreSQL (title, author, etc.)

Output is a JSON list of combined records.

    LOG_LEVEL=info \
      python src/tier1/tier1_5_query_lance.py "white hair" -k 10 --output tsv \
    | python src/tier1/tier1_5_lookup_events.py

"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

import lib.corpus_config as config
from lib.corpus_logging import logger
from lib.corpus_db import get_connection


def load_hits_tsv(path: Path | None) -> list[dict]:
    source = sys.stdin if path is None or str(path) == "-" else path.open()
    raw_lines = [ln.rstrip("\n\r") for ln in source]

    lines = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if set(s) <= {"-", "="}:
            continue
        if s.lower().startswith("encoding "):
            continue
        lines.append(ln)

    if not lines:
        return []

    first = lines[0]
    if "\t" in first:
        delim = "\t"
        header = [h.strip() for h in first.split(delim)]
        data_lines = lines[1:]
    else:
        delim = None
        header = re.split(r"\s{2,}", first.strip())
        data_lines = lines[1:]

    header = [h.strip().lstrip("#") for h in header]
    rename = {
        "distance": "_distance",
        "model": "embedding_model",
    }
    header = [rename.get(h, h) for h in header]

    if "event_id" not in header:
        raise SystemExit(
            "Could not find 'event_id' column in input.\n"
            f"Header parsed as: {header}\n"
            "Make sure you ran the query script with --output tsv "
            "(or --output json)."
        )

    rows = []
    for ln in data_lines:
        parts = ln.split(delim) if delim else re.split(r"\s{2,}", ln.strip())
        if len(parts) < len(header):
            continue
        row = dict(zip(header, parts))
        try:
            row["event_id"] = int(row["event_id"])
            row["year"] = int(row["year"])
            row["_distance"] = float(row["_distance"])
        except (KeyError, ValueError) as exc:
            logger.debug("Skipping malformed line: %r (%s)", ln, exc)
            continue
        rows.append(row)
    return rows


def load_hits_json(path: Path | None) -> list[dict]:
    source = sys.stdin if path is None or str(path) == "-" else path.open()
    data = json.load(source)
    for row in data:
        row["event_id"] = int(row["event_id"])
        row["year"] = int(row["year"])
        if "_distance" not in row and "distance" in row:
            row["_distance"] = row["distance"]
        row["_distance"] = float(row["_distance"])
    return data


def fetch_observations(
    store: Path,
    event_ids: list[int],
    years: list[int] | None = None,
) -> dict[int, dict]:
    dataset = ds.dataset(store, format="parquet", partitioning="hive")

    id_set = pa.array(event_ids, type=pa.uint64())
    filt = pc.is_in(ds.field("event_id"), value_set=id_set)
    if years:
        y_min, y_max = min(years), max(years)
        filt = filt & (ds.field("year") >= y_min) & (ds.field("year") <= y_max)

    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return {}

    drop = {c for c in table.column_names if c.startswith("emb_")}
    keep = [c for c in table.column_names if c not in drop]
    table = table.select(keep)

    out: dict[int, dict] = {}
    for batch in table.to_batches():
        for i in range(batch.num_rows):
            rec = {col: batch.column(col)[i].as_py() for col in keep}
            out[int(rec["event_id"])] = rec
    return out


def fetch_documents(doc_ids: list[str]) -> dict[str, dict]:
    """
    Look up document metadata from PostgreSQL for the given doc_ids.
    Returns {doc_id: {title, author, pub_year, ...}}.
    """
    if not doc_ids:
        return {}

    # de-dupe while preserving order
    unique = list(dict.fromkeys(doc_ids))

    sql = """
        SELECT
            doc_id,
            corpus,
            title,
            author,
            pub_year,
            publisher,
            pub_place,
            source_date_raw,
            token_count,
            lang,
            filepath
        FROM documents
        WHERE doc_id = ANY(%s)
    """

    out: dict[str, dict] = {}
    with get_connection(application_name="tier1_5_lookup") as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (unique,))
            cols = [d.name for d in cur.description]
            for row in cur.fetchall():
                rec = dict(zip(cols, row))
                out[str(rec["doc_id"])] = rec
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enrich Lance search hits with Parquet + Postgres metadata."
    )
    p.add_argument(
        "hits",
        type=Path,
        nargs="?",
        default=None,
        help="TSV/JSON file of hits (default: stdin).",
    )
    p.add_argument(
        "--format",
        choices=("tsv", "json"),
        default="tsv",
        help="Input format of the hits (default: tsv).",
    )
    p.add_argument(
        "--store",
        type=Path,
        default=config.EVENTSTORE_T1_PATH,
        help="Parquet event store root.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output.",
    )
    p.add_argument(
        "--no-docs",
        action="store_true",
        help="Skip PostgreSQL document lookup.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.format == "tsv":
        hits = load_hits_tsv(args.hits)
    else:
        hits = load_hits_json(args.hits)

    if not hits:
        print("[]")
        return

    event_ids = [h["event_id"] for h in hits]
    years = sorted({h["year"] for h in hits})

    logger.info(
        "Looking up %d event_ids (years %d–%d)…",
        len(event_ids),
        years[0],
        years[-1],
    )
    obs = fetch_observations(args.store, event_ids, years=years)

    # Collect doc_ids present in the observation rows
    doc_ids: list[str] = []
    for rec in obs.values():
        did = rec.get("doc_id")
        if did is not None:
            doc_ids.append(str(did))

    docs: dict[str, dict] = {}
    if doc_ids and not args.no_docs:
        logger.info("Looking up %d documents in PostgreSQL…", len(set(doc_ids)))
        docs = fetch_documents(doc_ids)
    elif not doc_ids and not args.no_docs:
        logger.warning(
            "No 'doc_id' column found in Parquet observations; "
            "document metadata will be omitted. "
            "Use --no-docs to silence this."
        )

    combined = []
    for h in hits:
        eid = h["event_id"]
        base = {
            "event_id": eid,
            "year": h["year"],
            "embedding_model": h.get("embedding_model"),
            "distance": h["_distance"],
        }
        meta = obs.get(eid)
        if meta is None:
            base["error"] = "event_id not found in store"
        else:
            base = {**meta, **base}
            did = meta.get("doc_id")
            if did is not None:
                doc = docs.get(str(did))
                if doc is not None:
                    base["document"] = doc
                else:
                    base["document"] = None
                    base["document_error"] = "doc_id not found in documents table"
        combined.append(base)

    indent = 2 if args.pretty else None
    print(json.dumps(combined, indent=indent, default=str))


if __name__ == "__main__":
    main()
