# tier1/tier1_corpus2events.py - formerly tier1/tier1_corpus2events.py.py
"""
Embed fixed windows of every document, at whichever scale --scale names
(local / medium / broad; sizes/strides come from tier1_seeds2events's
shared WINDOW_CONFIGS).

Local  → Lance (year-bucketed), via the shared tier1.vector_writer.VectorWriter
Colab  → Parquet on Google Drive, via ColabParquetWriter (same row schema)

Each (corpus, doc_id, scale) is tracked as its own row in embedding_jobs,
so running --scale medium and --scale broad over the same corpus are two
independent passes that don't mark each other's work done. Work is
claimed via embedding_jobs + FOR UPDATE SKIP LOCKED, filtered to one
scale per worker, so any number of workers (local + multiple Colab
notebooks, across scales) can run simultaneously without coordination.

Supports --dry-run (no writes, no job status changes).

Reset (a given scale's stuck jobs only):

    UPDATE embedding_jobs
    SET status      = 'pending',
        worker_id   = NULL,
        claimed_at  = NULL,
        finished_at = NULL,
        error       = NULL
    WHERE status IN ('running', 'failed')
      AND scale = 'medium';

"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import psycopg
from psycopg import Connection

from lib.corpus_config import (
    COLAB_MODE,
    OUT_DIR,
    LANCE_INDEXES_DIR,
    EMBED_BATCH_SIZE,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import get_macberth_embedder, normalize
from tier1.db_observation_backend import (
    allocate_event_ids,
    insert_events,
    create_events_table,
)
from tier1.tier1_seeds2events import WINDOW_CONFIGS, LANCE_MODEL_NAME, LANCE_BUCKET_SIZE
from tier1.vector_writer import VectorWriter, lance_table_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE_NAMES = tuple(config["name"] for config in WINDOW_CONFIGS)
DEFAULT_SCALE = "medium"

# window_id / window_token_pos are per-scale columns on `events`
# (see db_observation_backend.py). This is how a window pass for a given
# scale knows which pair of insert_events() kwargs to populate.
WINDOW_ID_FIELDS = {
    config["name"]: (
        f"{config['name']}_window_id",
        f"{config['name']}_window_token_pos",
    )
    for config in WINDOW_CONFIGS
}

PARQUET_DIR = OUT_DIR / "macberth_windows"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)


def resolve_window_config(scale: str) -> tuple[int, int]:
    """Look up (size, stride) for a scale name from the shared WINDOW_CONFIGS."""
    try:
        config = next(c for c in WINDOW_CONFIGS if c["name"] == scale)
    except StopIteration:
        raise ValueError(
            f"Unknown scale {scale!r}; choices are {list(SCALE_NAMES)}"
        )
    return config["size"], config["stride"]

#
# For Windows lack of uname support
#
def _default_worker_id() -> str:
    try:
        host = os.uname().nodename          # Unix
    except AttributeError:
        host = os.environ.get("COMPUTERNAME", "windows")  # Windows
    return f"{host}-{os.getpid()}"


# ---------------------------------------------------------------------------
# Jobs table
# ---------------------------------------------------------------------------

def ensure_jobs_table(conn: Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                job_id      BIGSERIAL PRIMARY KEY,
                corpus      TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                scale       TEXT NOT NULL DEFAULT 'medium',
                status      TEXT NOT NULL DEFAULT 'pending',
                worker_id   TEXT,
                claimed_at  TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                error       TEXT
            );
        """)
        conn.commit()

    _migrate_jobs_table_add_scale(conn)

    with conn.cursor() as cur:
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_embedding_jobs_corpus_doc_scale
                ON embedding_jobs (corpus, doc_id, scale);
            CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status
                ON embedding_jobs (status);
            CREATE INDEX IF NOT EXISTS idx_embedding_jobs_scale
                ON embedding_jobs (scale);
        """)
        conn.commit()


def _migrate_jobs_table_add_scale(conn: Connection) -> None:
    """
    One-time migration for embedding_jobs tables created before `scale`
    existed (when this pipeline only ever ran a single medium-window
    pass). Every statement here is idempotent, so it's safe to call on
    every startup: a table that already has `scale` NOT NULL and no old
    (corpus, doc_id) constraint is a no-op.

    Existing rows are backfilled as scale='medium' because that was the
    only pass this table ever tracked historically — not a guess.
    """
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE embedding_jobs
                ADD COLUMN IF NOT EXISTS scale TEXT;
        """)
        cur.execute("""
            UPDATE embedding_jobs SET scale = 'medium' WHERE scale IS NULL;
        """)
        cur.execute("""
            ALTER TABLE embedding_jobs
                ALTER COLUMN scale SET DEFAULT 'medium';
        """)
        cur.execute("""
            ALTER TABLE embedding_jobs
                ALTER COLUMN scale SET NOT NULL;
        """)
        # The old schema's inline UNIQUE (corpus, doc_id) gets Postgres's
        # default constraint name below. Drop it so a document can now
        # have one job per scale instead of one job total.
        cur.execute("""
            ALTER TABLE embedding_jobs
                DROP CONSTRAINT IF EXISTS embedding_jobs_corpus_doc_id_key;
        """)
    conn.commit()


def populate_jobs(
    conn: Connection,
    *,
    scale: str,
    corpus: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> int:
    """Insert one (corpus, doc_id, scale) job per document not already present."""
    clauses = ["TRUE"]
    params: list = []

    if corpus is not None:
        clauses.append("d.corpus = %s")
        params.append(corpus)
    if min_year is not None:
        clauses.append("d.pub_year >= %s")
        params.append(min_year)
    if max_year is not None:
        clauses.append("d.pub_year <= %s")
        params.append(max_year)

    where = " AND ".join(clauses)

    with conn.cursor() as cur:
        cur.execute(f"""
            INSERT INTO embedding_jobs (corpus, doc_id, scale)
            SELECT d.corpus, d.doc_id, %s
            FROM documents d
            WHERE {where}
            ON CONFLICT (corpus, doc_id, scale) DO NOTHING
        """, [scale, *params])
        conn.commit()
        return cur.rowcount


def claim_job(
    conn: Connection,
    worker_id: str,
    *,
    scale: str,
    dry_run: bool = False,
) -> Optional[tuple[int, str, str]]:
    """Claim one pending job for this scale. Returns (job_id, corpus, doc_id) or None."""
    if dry_run:
        # In dry-run we still want to see real pending work,
        # but we never mark it running.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT job_id, corpus, doc_id
                FROM embedding_jobs
                WHERE status = 'pending' AND scale = %s
                ORDER BY job_id
                LIMIT 1
            """, (scale,))
            return cur.fetchone()

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE embedding_jobs
            SET status = 'running',
                worker_id = %s,
                claimed_at = now()
            WHERE job_id = (
                SELECT job_id
                FROM embedding_jobs
                WHERE status = 'pending' AND scale = %s
                ORDER BY job_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING job_id, corpus, doc_id
        """, (worker_id, scale))
        row = cur.fetchone()
    conn.commit()
    return row


def mark_job_done(conn: Connection, job_id: int, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE embedding_jobs
            SET status = 'done', finished_at = now(), error = NULL
            WHERE job_id = %s
        """, (job_id,))
    conn.commit()


def mark_job_failed(
    conn: Connection,
    job_id: int,
    error: str,
    *,
    dry_run: bool = False,
) -> None:
    if dry_run:
        return
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE embedding_jobs
            SET status = 'failed', finished_at = now(), error = %s
            WHERE job_id = %s
        """, (error[:2000], job_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Window production (streaming – safe for Bibles)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Window:
    start_idx: int
    end_idx: int
    text: str


def iter_windows(
    conn: Connection,
    corpus: str,
    doc_id: str,
    *,
    size: int,
    stride: int,
) -> Iterator[tuple[Optional[int], Window]]:
    """
    Yield (pub_year, Window) without ever holding the whole document.
    Uses fetchmany so memory stays bounded even for Bibles.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.token, d.pub_year
            FROM tokens AS t
            JOIN documents AS d
                ON d.corpus = t.corpus
                AND d.doc_id = t.doc_id
            WHERE t.corpus = %s
              AND t.doc_id = %s
            ORDER BY t.token_idx
        """, (corpus, doc_id))

        buffer: list[str] = []
        pub_year: Optional[int] = None
        start_of_window = 0
        FETCH = 4096

        while True:
            rows = cur.fetchmany(FETCH)
            if not rows:
                break

            for token, year in rows:
                if pub_year is None:
                    pub_year = year
                buffer.append(token)

                while len(buffer) >= size:
                    text = " ".join(buffer[:size])
                    yield pub_year, Window(
                        start_idx=start_of_window,
                        end_idx=start_of_window + size,
                        text=text,
                    )
                    del buffer[:stride]
                    start_of_window += stride

        # final partial window
        if buffer:
            text = " ".join(buffer)
            yield pub_year, Window(
                start_idx=start_of_window,
                end_idx=start_of_window + len(buffer),
                text=text,
            )

# ---------------------------------------------------------------------------
# Vector writers
#
# Local runs use the shared tier1.vector_writer.VectorWriter (Lance),
# the exact same class tier1_seeds2events.py uses — same lean schema
# (event_id, pub_year, embedding_model, vector), same table naming, same
# dedup-on-write, same index build. Colab has no Lance available, so it
# gets a Parquet-writing counterpart with an identical interface and the
# same row shape, so a later local ingest of the shards needs no
# transform, just `table.add(pq.read_table(shard).to_pylist())`.
# ---------------------------------------------------------------------------

class ColabParquetWriter:
    """Same interface and row schema as VectorWriter, but shards to Parquet."""

    def __init__(
        self,
        out_dir: Path,
        *,
        scale: str,
        model_name: str,
        bucket_size: int,
        dry_run: bool = False,
    ) -> None:
        self.out_dir = out_dir
        self.scale = scale
        self.model_name = model_name
        self.bucket_size = bucket_size
        self.dry_run = dry_run

    def table_name(self, pub_year: int) -> str:
        return lance_table_name(self.scale, self.model_name, pub_year, self.bucket_size)

    def write(self, *, event_ids: list[int], pub_year: Optional[int], vectors: np.ndarray) -> int:
        n = len(event_ids)
        if n == 0:
            return 0
        if pub_year is None:
            raise ValueError("pub_year is required to bucket a Parquet write")

        rows = [
            {
                "event_id": int(event_ids[i]),
                "pub_year": int(pub_year),
                "embedding_model": self.model_name,
                "vector": np.asarray(vectors[i], dtype=np.float32).tolist(),
            }
            for i in range(n)
        ]

        table_name = self.table_name(pub_year)

        if self.dry_run:
            logger.info(
                "[dry-run] would write %d rows to Parquet shard for %s",
                n, table_name,
            )
            return n

        shard_dir = self.out_dir / table_name
        shard_dir.mkdir(parents=True, exist_ok=True)
        fname = shard_dir / f"{uuid.uuid4().hex}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), fname, compression="zstd")
        return n


def make_vector_writer(
    scale: str,
    lance_root: Path,
    *,
    dry_run: bool = False,
):
    """Pick the Lance writer locally, or the Parquet counterpart on Colab."""
    if COLAB_MODE:
        return ColabParquetWriter(
            PARQUET_DIR,
            scale=scale,
            model_name=LANCE_MODEL_NAME,
            bucket_size=LANCE_BUCKET_SIZE,
            dry_run=dry_run,
        )

    return VectorWriter(
        lance_root,
        scale=scale,
        model_name=LANCE_MODEL_NAME,
        bucket_size=LANCE_BUCKET_SIZE,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _flush_batch(
    conn: Connection,
    embedder,
    writer,
    scale: str,
    corpus: str,
    doc_id: str,
    pub_year: Optional[int],
    windows: list[Window],
    *,
    dry_run: bool = False,
) -> int:
    texts = [w.text for w in windows]
    raw = embedder.encode(texts, convert_to_numpy=True)
    vectors = np.stack([
        normalize(v) if (n := normalize(v)) is not None else v
        for v in raw
    ]).astype(np.float32)

    if dry_run:
        # Still run the model so we can measure time / catch OOM,
        # but allocate no IDs and write nothing.
        logger.info(
            "[dry-run] embedded %d %s windows for %s/%s (year=%s)",
            len(windows), scale, corpus, doc_id, pub_year,
        )
        return len(windows)

    event_ids = allocate_event_ids(conn, len(windows))

    window_id_field, window_pos_field = WINDOW_ID_FIELDS[scale]

    insert_events(
        conn,
        event_id=event_ids,
        corpus=[corpus] * len(windows),
        doc_id=[doc_id] * len(windows),
        token=["[WINDOW]"] * len(windows),
        token_idx=[w.start_idx for w in windows],
        pub_year=[pub_year] * len(windows),
        **{
            window_id_field: [w.start_idx for w in windows],
            window_pos_field: [0] * len(windows),
        },
    )
    conn.commit()

    # Postgres owns provenance (corpus/doc_id/window bounds live there via
    # the columns above); Lance/Parquet only need event_id + pub_year to
    # bucket and dedup the vector itself.
    return writer.write(
        event_ids=event_ids,
        pub_year=pub_year,
        vectors=vectors,
    )


def process_document(
    conn: Connection,
    embedder,
    writer,
    scale: str,
    corpus: str,
    doc_id: str,
    *,
    window_size: int,
    window_stride: int,
    dry_run: bool = False,
) -> int:
    total_written = 0
    batch_windows: list[Window] = []
    batch_year: Optional[int] = None

    for pub_year, window in iter_windows(
        conn, corpus, doc_id, size=window_size, stride=window_stride
    ):
        if batch_year is None:
            batch_year = pub_year

        batch_windows.append(window)

        if len(batch_windows) >= EMBED_BATCH_SIZE:
            total_written += _flush_batch(
                conn, embedder, writer, scale,
                corpus, doc_id, batch_year, batch_windows,
                dry_run=dry_run,
            )
            batch_windows = []

    if batch_windows:
        total_written += _flush_batch(
            conn, embedder, writer, scale,
            corpus, doc_id, batch_year, batch_windows,
            dry_run=dry_run,
        )

    return total_written


def run_worker(
    *,
    scale: str = DEFAULT_SCALE,
    worker_id: Optional[str] = None,
    max_docs: Optional[int] = None,
    backend: str = "onnx",
    dry_run: bool = False,
) -> None:
    worker_id = worker_id or _default_worker_id()
    window_size, window_stride = resolve_window_config(scale)

    logger.info(
        "[tier1_corpus2events.py] worker %s starting scale=%s size=%d stride=%d "
        "(COLAB_MODE=%s, dry_run=%s)",
        worker_id, scale, window_size, window_stride, COLAB_MODE, dry_run,
    )

    conn = get_connection()
    backend = None if backend == "auto" else backend
    embedder = get_macberth_embedder(pooling="mean", backend=backend)
    writer = make_vector_writer(scale, LANCE_INDEXES_DIR, dry_run=dry_run)

    processed = 0
    while True:
        if max_docs is not None and processed >= max_docs:
            break

        job = claim_job(conn, worker_id, scale=scale, dry_run=dry_run)
        if job is None:
            logger.info("[tier1_corpus2events.py] no more pending %s jobs", scale)
            break

        job_id, corpus, doc_id = job
        started = time.perf_counter()

        try:
            written = process_document(
                conn, embedder, writer, scale, corpus, doc_id,
                window_size=window_size,
                window_stride=window_stride,
                dry_run=dry_run,
            )
            mark_job_done(conn, job_id, dry_run=dry_run)
            elapsed = time.perf_counter() - started
            logger.info(
                "[tier1_corpus2events.py] %s %s/%s scale=%s windows=%d  %.1fs",
                "dry-run" if dry_run else "done",
                corpus, doc_id, scale, written, elapsed,
            )
        except Exception as exc:
            logger.exception("[window_embedder] failed %s/%s (scale=%s)", corpus, doc_id, scale)
            try:
                conn.rollback()          # ← clear the aborted transaction
            except Exception:
                pass
            mark_job_failed(conn, job_id, str(exc), dry_run=dry_run)

        processed += 1

        # In dry-run we only ever look at the first pending job
        # (otherwise we would loop forever on the same row).
        if dry_run:
            break

    conn.close()
    logger.info(
        "[tier1_corpus2events.py] worker %s finished scale=%s (%d documents)",
        worker_id, scale, processed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed-window MacBERTh embedder (local/medium/broad)")
    p.add_argument("--populate", action="store_true", help="Populate embedding_jobs from documents table")
    p.add_argument(
        "--scale",
        choices=SCALE_NAMES,
        default=DEFAULT_SCALE,
        help=f"Which window scale to run (default: {DEFAULT_SCALE})",
    )
    p.add_argument("--corpus", default=None)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--worker-id", default=None)
    p.add_argument("--max-docs", type=int, default=None, help="Stop after this many documents (useful for testing)")
    p.add_argument("--dry-run", action="store_true", help="Run embedding but write nothing and leave jobs untouched")
    p.add_argument(
        "--backend",
        choices=("auto", "onnx", "pytorch"),
        default="auto",
        help="Embedding backend (default: auto = fastest available)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conn = get_connection()

    if args.populate:
        logger.info("[tier1_corpus2events.py] Shall ensure jobs table")
        ensure_jobs_table(conn)
        logger.info("[tier1_corpus2events.py] Shall populate %s jobs", args.scale)
        n = populate_jobs(
            conn,
            scale=args.scale,
            corpus=args.corpus,
            min_year=args.min_year,
            max_year=args.max_year,
        )
        logger.info("[tier1_corpus2events.py] populated %d %s jobs", n, args.scale)
        conn.close()
        return

    conn.close()

    run_worker(
        scale=args.scale,
        worker_id=args.worker_id,
        max_docs=args.max_docs,
        backend=args.backend,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

