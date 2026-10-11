"""
colab_export_for.py

Run ONCE, locally, where Postgres lives.

Exports one row per document (corpus, doc_id, pub_year, token_idx[], tokens[])
into parquet shards of DOCS_PER_SHARD documents. Upload EMB_OUTPUT_DIR to Drive.

To export the full corpus later, change TABLE / DOC_TABLE.
"""
from itertools import groupby
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lib.corpus_db import get_connection
from lib.corpus_config import OUT_DIR
from lib.corpus_logging import logger

# Local filesystem output directory
EMB_OUTPUT_DIR = OUT_DIR / Path("export/tokens")

# PostgreSQL tables
TABLE = "pamphlet_tokens"
DOC_TABLE = "pamphlet_corpus"

DOCS_PER_SHARD = 500

SCHEMA = pa.schema([
    ("corpus", pa.string()),
    ("doc_id", pa.string()),
    ("pub_year", pa.int32()),
    ("token_idx", pa.list_(pa.int32())),
    ("tokens", pa.list_(pa.string())),
])


def flush(docs: list[dict], shard_no: int) -> None:
    table = pa.Table.from_pylist(docs, schema=SCHEMA)
    final = EMB_OUTPUT_DIR / f"tokens_{shard_no:05d}.parquet"
    tmp = final.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(final)
    logger.info(f"wrote {final} ({len(docs)} docs)")


def main() -> None:
    EMB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(application_name="export-tokens")

    # Named (server-side) cursor: streams instead of loading everything.
    with conn.cursor(name="export_tokens") as cur:
        cur.itersize = 100_000
        cur.execute(
            f"""
            SELECT t.corpus, t.doc_id, d.pub_year, t.token_idx, t.token
            FROM {TABLE} AS t
            JOIN {DOC_TABLE} AS d
              ON d.corpus = t.corpus AND d.doc_id = t.doc_id
            ORDER BY t.corpus, t.doc_id, t.token_idx
            """
        )

        docs: list[dict] = []
        shard_no = 0

        for (corpus, doc_id), rows in groupby(cur, key=lambda r: (r[0], r[1])):
            logger.info(f"Do {doc_id}")
            rows = list(rows)
            docs.append({
                "corpus": corpus,
                "doc_id": doc_id,
                "pub_year": rows[0][2],
                "token_idx": [r[3] for r in rows],
                "tokens": [r[4] for r in rows],
            })

            if len(docs) >= DOCS_PER_SHARD:
                flush(docs, shard_no)
                docs, shard_no = [], shard_no + 1

        if docs:
            flush(docs, shard_no)

    conn.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
