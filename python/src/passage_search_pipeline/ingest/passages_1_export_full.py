"""
passages_1_export_full.py

Exports one row per document (corpus, doc_id, pub_year, token_idx[], tokens[])
into parquet shards of --docs-per-shard documents. Upload the output dir to Drive.

By default, documents are skipped if they are:
  * already present in any passage Lance table (--lance-dir), or
  * already present in a parquet shard in the output dir (re-run safety).

If --force-doc-id is given, ONLY those doc_ids are exported, regardless of
whether they are in Lance, in existing shards, or in CORPUS_SQL. No other
documents are processed in that mode.

New shards are numbered after the highest existing shard, so reruns never
overwrite earlier output (use --overwrite to wipe and start from 0).

"""
import argparse
import re
from itertools import groupby
from pathlib import Path

import lancedb
import pyarrow as pa
import pyarrow.parquet as pq

from lib.corpus_db import get_connection
from lib.corpus_config import MAPPED_GOOGLE_DRIVE, OUT_DIR, LANCE_INDEXES_DIR
from lib.corpus_logging import logger

from lib.corpus_albinism import CORPUS_SQL

# Defaults (all overridable from the CLI)
DEFAULT_OUTPUT_DIR = MAPPED_GOOGLE_DRIVE / Path("souroce_tokens")
DEFAULT_LANCE_DIR = LANCE_INDEXES_DIR
DEFAULT_LANCE_PREFIXES = ("passage_l8__", "passage_last__", "passage_mean4__")
DEFAULT_DOCS_PER_SHARD = 10  # 500

SHARD_PREFIX = "tokens_"
SHARD_RE = re.compile(rf"^{re.escape(SHARD_PREFIX)}(\d+)\.parquet$")

# PostgreSQL tables
DOC_TABLE = "documents"
TABLE = "tokens"

SCHEMA = pa.schema([
    ("corpus", pa.string()),
    ("doc_id", pa.string()),
    ("pub_year", pa.int32()),
    ("token_idx", pa.list_(pa.int32())),
    ("tokens", pa.list_(pa.string())),
])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export full-document tokens to parquet shards, skipping documents already indexed in Lance." )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for parquet shards (default: %(default)s)")
    p.add_argument("--lance-dir", type=Path, default=DEFAULT_LANCE_DIR, help="LanceDB directory to read indexed docs from " "(default: %(default)s)")
    p.add_argument("--lance-prefix", action="append", dest="lance_prefixes",
                   metavar="PREFIX",
                   help="Lance table-name prefix to treat as 'indexed'. Repeatable. Default: " + ", ".join(DEFAULT_LANCE_PREFIXES))
    p.add_argument("--docs-per-shard", type=int, default=DEFAULT_DOCS_PER_SHARD, help="Documents per parquet shard (default: %(default)s)")
    p.add_argument("--force-doc-id", action="append", default=[],
                   dest="force_doc_ids", metavar="DOC_ID",
                   help="Export ONLY this doc_id, ignoring Lance, existing shards and the corpus filter. "
                        "Repeatable; when used, no other documents are exported.")
    p.add_argument("--no-skip-indexed", action="store_true", help="Do not skip documents found in Lance")
    p.add_argument("--no-skip-exported", action="store_true", help="Do not skip documents already in existing parquet shards")
    p.add_argument("--overwrite", action="store_true", help="Delete existing shards in the output dir and start numbering at 0 (implies --no-skip-exported)")
    p.add_argument("--limit-docs", type=int, default=None, help="Stop after exporting this many documents (for testing)")
    p.add_argument("--itersize", type=int, default=100_000, help="Server-side cursor fetch size (default: %(default)s)")
    args = p.parse_args()

    if args.docs_per_shard < 1:
        p.error("--docs-per-shard must be >= 1")
    if args.force_doc_ids and args.overwrite:
        p.error("--force-doc-id cannot be combined with --overwrite "
                "(it would delete existing shards before a targeted export)")
    if args.lance_prefixes is None:
        args.lance_prefixes = list(DEFAULT_LANCE_PREFIXES)
    return args


# --------------------------------------------------------------------------
# Skip-set sources
# --------------------------------------------------------------------------
def indexed_documents(lance_dir: Path, prefixes: list[str]) -> set[tuple[str, str]]:
    """(corpus, doc_id) pairs present in any matching Lance passage table."""
    indexed: set[tuple[str, str]] = set()
    if not lance_dir.exists():
        logger.warning("Lance dir %s does not exist; treating as empty", lance_dir)
        return indexed

    db = lancedb.connect(str(lance_dir))
    names = [n for n in db.list_tables().tables if n.startswith(tuple(prefixes))]

    for name in names:
        rows = db.open_table(name).search().select(["corpus", "doc_id"]).to_arrow()
        indexed.update(zip(rows["corpus"].to_pylist(), rows["doc_id"].to_pylist()))

    logger.info("Found %d indexed documents across %d Lance tables",
                len(indexed), len(names))
    return indexed


def existing_shards(output_dir: Path) -> list[tuple[int, Path]]:
    shards = []
    for path in output_dir.iterdir():
        m = SHARD_RE.match(path.name)
        if m:
            shards.append((int(m.group(1)), path))
    return sorted(shards)


def exported_documents(shards: list[tuple[int, Path]]) -> set[tuple[str, str]]:
    """(corpus, doc_id) pairs already written to parquet shards."""
    done: set[tuple[str, str]] = set()
    for _, path in shards:
        t = pq.read_table(path, columns=["corpus", "doc_id"])
        done.update(zip(t["corpus"].to_pylist(), t["doc_id"].to_pylist()))
    logger.info("Found %d documents already exported in %d shards",
                len(done), len(shards))
    return done


# --------------------------------------------------------------------------
# Postgres
# --------------------------------------------------------------------------
def populate_skip_docs(conn, skip: set[tuple[str, str]]) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE export_skip_docs (
                corpus text NOT NULL,
                doc_id text NOT NULL,
                PRIMARY KEY (corpus, doc_id)
            ) ON COMMIT PRESERVE ROWS
        """)
        ordered = sorted(skip)
        for offset in range(0, len(ordered), 5000):
            cur.executemany(
                """
                INSERT INTO export_skip_docs (corpus, doc_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
                """,
                ordered[offset:offset + 5000],
            )
    conn.commit()


def populate_force_docs(conn, doc_ids: set[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE export_force_docs (
                doc_id text PRIMARY KEY
            ) ON COMMIT PRESERVE ROWS
        """)
        cur.executemany(
            "INSERT INTO export_force_docs (doc_id) VALUES (%s) ON CONFLICT DO NOTHING",
            [(d,) for d in sorted(doc_ids)],
        )
    conn.commit()


def build_query(force_mode: bool) -> str:
    # No bind parameters, so any literal % inside CORPUS_SQL keeps working.
    if force_mode:
        # Only the forced doc_ids: ignore CORPUS_SQL and all skip sets.
        ctes = f"""
        eligible_docs AS (
            SELECT d.corpus, d.doc_id
            FROM {DOC_TABLE} AS d
            JOIN export_force_docs AS f ON f.doc_id = d.doc_id
        )"""
    else:
        ctes = f"""
        selected_docs AS (
            {CORPUS_SQL}
        ),
        eligible_docs AS (
            SELECT s.corpus, s.doc_id
            FROM selected_docs AS s
            WHERE NOT EXISTS (
                SELECT 1
                FROM export_skip_docs AS x
                WHERE x.corpus = s.corpus
                  AND x.doc_id = s.doc_id
            )
        )"""

    return f"""
        WITH {ctes}
        SELECT
            t.corpus,
            t.doc_id,
            d.pub_year,
            t.token_idx,
            t.token
        FROM {TABLE} AS t
        JOIN {DOC_TABLE} AS d
          ON d.corpus = t.corpus
         AND d.doc_id = t.doc_id
        JOIN eligible_docs AS e
          ON e.corpus = t.corpus
         AND e.doc_id = t.doc_id
        ORDER BY t.corpus, t.doc_id, t.token_idx
    """


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def flush(docs: list[dict], shard_no: int, output_dir: Path) -> None:
    table = pa.Table.from_pylist(docs, schema=SCHEMA)
    final = output_dir / f"{SHARD_PREFIX}{shard_no:05d}.parquet"
    tmp = final.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(final)
    logger.info("Wrote %s (%d docs)", final, len(docs))


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    force = set(args.force_doc_ids)

    # Shard bookkeeping
    shards = existing_shards(output_dir)
    if args.overwrite:
        for _, path in shards:
            path.unlink()
        logger.info("Deleted %d existing shards", len(shards))
        shards = []
    shard_no = shards[-1][0] + 1 if shards else 0

    # Build skip set (not used at all in force mode)
    skip: set[tuple[str, str]] = set()
    if force:
        logger.info("Force mode: exporting ONLY doc_ids %s "
                    "(ignoring Lance, existing shards, and CORPUS_SQL); "
                    "first new shard is %05d", sorted(force), shard_no)
    else:
        if not args.no_skip_indexed:
            skip |= indexed_documents(args.lance_dir, args.lance_prefixes)
        if not args.no_skip_exported and shards:
            skip |= exported_documents(shards)
        logger.info("Skipping %d documents; first new shard is %05d",
                    len(skip), shard_no)

    conn = get_connection(application_name="export-tokens")
    try:
        if force:
            populate_force_docs(conn, force)
        else:
            populate_skip_docs(conn, skip)

        docs: list[dict] = []
        exported = 0
        seen_ids: set[str] = set()

        with conn.cursor(name="export_tokens") as cur:
            cur.itersize = args.itersize
            cur.execute(build_query(force_mode=bool(force)))

            for (corpus, doc_id), grouped in groupby(cur, key=lambda r: (r[0], r[1])):
                rows = list(grouped)
                seen_ids.add(doc_id)
                docs.append({
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "pub_year": rows[0][2],
                    "token_idx": [r[3] for r in rows],
                    "tokens": [r[4] for r in rows],
                })
                exported += 1
                if exported % 100 == 0:
                    logger.info("Exported %d documents so far", exported)

                if len(docs) >= args.docs_per_shard:
                    flush(docs, shard_no, output_dir)
                    docs, shard_no = [], shard_no + 1

                if args.limit_docs and exported >= args.limit_docs:
                    logger.info("Reached --limit-docs=%d", args.limit_docs)
                    break

            if docs:
                flush(docs, shard_no, output_dir)

        if force:
            missing = force - seen_ids
            if missing:
                logger.warning("Forced doc_ids not found in %s/%s: %s",
                               DOC_TABLE, TABLE, sorted(missing))

        logger.info("Done: exported %d documents", exported)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
