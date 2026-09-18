# tier1/tier1_corpus2events.py - formerly tier1/tier1_corpus2events.py.py
"""
Embed fixed medium windows of every document.

Local  → Lance (year-bucketed)
Colab  → Parquet on Google Drive

Work is claimed via embedding_jobs + FOR UPDATE SKIP LOCKED
so any number of workers (local + multiple Colab notebooks) can run
simultaneously without coordination.

Supports --dry-run (no writes, no job status changes).
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
import lancedb
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
from tier1.tier1_seeds2events import (
    WINDOW_CONFIGS,
    ACTIVE_SCALES,
    LANCE_MODEL_NAME,
    LANCE_BUCKET_SIZE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALE = "medium"

_WINDOW_CONFIG = next(
    config for config in WINDOW_CONFIGS
    if config["name"] == SCALE
)

WINDOW_SIZE = _WINDOW_CONFIG["size"]
WINDOW_STRIDE = _WINDOW_CONFIG["stride"]

PARQUET_DIR = OUT_DIR / "macberth_windows"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

logger.info(
    "[tier1_corpus2events.py] scale=%s size=%d stride=%d; parquet dir=%s",
    SCALE, WINDOW_SIZE, WINDOW_STRIDE, PARQUET_DIR
)

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
                status      TEXT NOT NULL DEFAULT 'pending',
                worker_id   TEXT,
                claimed_at  TIMESTAMPTZ,
                finished_at TIMESTAMPTZ,
                error       TEXT,
                UNIQUE (corpus, doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status
                ON embedding_jobs (status);
        """)
        conn.commit()


def populate_jobs(
    conn: Connection,
    *,
    corpus: Optional[str] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> int:
    """Insert one job per document that is not already present."""
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
            INSERT INTO embedding_jobs (corpus, doc_id)
            SELECT d.corpus, d.doc_id
            FROM documents d
            WHERE {where}
            ON CONFLICT (corpus, doc_id) DO NOTHING
        """, params)
        conn.commit()
        return cur.rowcount


def claim_job(
    conn: Connection,
    worker_id: str,
    *,
    dry_run: bool = False,
) -> Optional[tuple[int, str, str]]:
    """Claim one pending job. Returns (job_id, corpus, doc_id) or None."""
    if dry_run:
        # In dry-run we still want to see real pending work,
        # but we never mark it running.
        with conn.cursor() as cur:
            cur.execute("""
                SELECT job_id, corpus, doc_id
                FROM embedding_jobs
                WHERE status = 'pending'
                ORDER BY job_id
                LIMIT 1
            """)
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
                WHERE status = 'pending'
                ORDER BY job_id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING job_id, corpus, doc_id
        """, (worker_id,))
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
    size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> Iterator[tuple[Optional[int], Window]]:
    """
    Yield (pub_year, Window) without ever holding the whole document.
    Uses a server-side cursor so Postgres streams the tokens.
    """
    with conn.cursor(name=f"win_{corpus}_{doc_id}") as cur:  # server-side
        cur.itersize = 4096
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

        for token, year in cur:
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

        # final partial window (keep for now)
        if buffer:
            text = " ".join(buffer)
            yield pub_year, Window(
                start_idx=start_of_window,
                end_idx=start_of_window + len(buffer),
                text=text,
            )


# ---------------------------------------------------------------------------
# Lance / Parquet writers
# ---------------------------------------------------------------------------

def year_bucket(year: int) -> tuple[int, int]:
    start = (year // LANCE_BUCKET_SIZE) * LANCE_BUCKET_SIZE
    return start, start + LANCE_BUCKET_SIZE - 1


def lance_table_name(year: int) -> str:
    start, end = year_bucket(year)
    return f"{SCALE}__{LANCE_MODEL_NAME}__{start:04d}_{end:04d}"


class VectorWriter:
    """Writes the same row schema to Lance (local) or Parquet (Colab)."""

    def __init__(self, lance_root: Path, *, dry_run: bool = False):
        self.colab = COLAB_MODE
        self.lance_root = lance_root
        self.dry_run = dry_run
        self.tables: dict[str, object] = {}
        if not self.colab and not self.dry_run:
            self.db = lancedb.connect(str(lance_root))

    def write(
        self,
        *,
        event_ids: list[int],
        corpus: str,
        doc_id: str,
        pub_year: Optional[int],
        windows: list[Window],
        vectors: np.ndarray,
    ) -> int:
        if pub_year is None:
            raise ValueError(f"Document {corpus}/{doc_id} has no pub_year")

        n = len(event_ids)
        assert n == len(windows) == len(vectors)

        rows = []
        for i in range(n):
            rows.append({
                "event_id": int(event_ids[i]),
                "corpus": corpus,
                "doc_id": doc_id,
                "window_id": f"{doc_id}:{windows[i].start_idx:06d}",
                "start_idx": windows[i].start_idx,
                "end_idx": windows[i].end_idx,
                "pub_year": pub_year,
                "embedding_model": LANCE_MODEL_NAME,
                "vector": vectors[i].astype(np.float32).tolist(),
            })

        if self.dry_run:
            logger.info(
                "[dry-run] would write %d rows for %s/%s → %s",
                n, corpus, doc_id, lance_table_name(pub_year),
            )
            return n

        if self.colab:
            return self._write_parquet(rows, pub_year)
        else:
            return self._write_lance(rows, pub_year)

    def _write_parquet(self, rows: list[dict], pub_year: int) -> int:
        table_name = lance_table_name(pub_year)
        out_dir = PARQUET_DIR / table_name
        out_dir.mkdir(parents=True, exist_ok=True)

        fname = out_dir / f"{uuid.uuid4().hex}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, fname, compression="zstd")
        return len(rows)

    def _write_lance(self, rows: list[dict], pub_year: int) -> int:
        table_name = lance_table_name(pub_year)
        table = self._open_table(table_name, dim=len(rows[0]["vector"]))

        existing = set()
        if table.count_rows() > 0:
            ids = [r["event_id"] for r in rows]
            arrow = table.to_arrow()
            existing = set(arrow.column("event_id").to_pylist()) & set(ids)

        new_rows = [r for r in rows if r["event_id"] not in existing]
        if new_rows:
            table.add(new_rows, mode="append")
        return len(new_rows)

    def _open_table(self, name: str, dim: int):
        if name in self.tables:
            return self.tables[name]

        existing = set(self.db.list_tables().tables)
        if name in existing:
            table = self.db.open_table(name)
        else:
            logger.info("[tier1_corpus2events.py] creating Lance table %s", name)
            schema = pa.schema([
                pa.field("event_id", pa.uint64()),
                pa.field("corpus", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("window_id", pa.string()),
                pa.field("start_idx", pa.int32()),
                pa.field("end_idx", pa.int32()),
                pa.field("pub_year", pa.int32()),
                pa.field("embedding_model", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ])
            table = self.db.create_table(name, schema=schema)

        self.tables[name] = table
        return table


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def _flush_batch(
    conn: Connection,
    embedder,
    writer: VectorWriter,
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
            "[dry-run] embedded %d windows for %s/%s (year=%s)",
            len(windows), corpus, doc_id, pub_year,
        )
        return len(windows)

    event_ids = allocate_event_ids(conn, len(windows))

    insert_events(
        conn,
        event_id=event_ids,
        corpus=[corpus] * len(windows),
        doc_id=[doc_id] * len(windows),
        token=["[WINDOW]"] * len(windows),
        token_idx=[w.start_idx for w in windows],
        pub_year=[pub_year] * len(windows),
        medium_window_id=[w.start_idx for w in windows],
        medium_window_token_pos=[0] * len(windows),
    )
    conn.commit()

    return writer.write(
        event_ids=event_ids,
        corpus=corpus,
        doc_id=doc_id,
        pub_year=pub_year,
        windows=windows,
        vectors=vectors,
    )


def process_document(
    conn: Connection,
    embedder,
    writer: VectorWriter,
    corpus: str,
    doc_id: str,
    *,
    dry_run: bool = False,
) -> int:
    total_written = 0
    batch_windows: list[Window] = []
    batch_year: Optional[int] = None

    for pub_year, window in iter_windows(conn, corpus, doc_id):
        if batch_year is None:
            batch_year = pub_year

        batch_windows.append(window)

        if len(batch_windows) >= EMBED_BATCH_SIZE:
            total_written += _flush_batch(
                conn, embedder, writer,
                corpus, doc_id, batch_year, batch_windows,
                dry_run=dry_run,
            )
            batch_windows = []

    if batch_windows:
        total_written += _flush_batch(
            conn, embedder, writer,
            corpus, doc_id, batch_year, batch_windows,
            dry_run=dry_run,
        )

    return total_written


def run_worker(
    *,
    worker_id: Optional[str] = None,
    max_docs: Optional[int] = None,
    backend: str = "onnx",
    dry_run: bool = False,
) -> None:
    worker_id = worker_id or _default_worker_id()
    logger.info(
        "[tier1_corpus2events.py] worker %s starting (COLAB_MODE=%s, dry_run=%s)",
        worker_id, COLAB_MODE, dry_run,
    )

    conn = get_connection()
    backend = None if backend == "auto" else backend
    embedder = get_macberth_embedder(pooling="mean", backend=backend)
    writer = VectorWriter(LANCE_INDEXES_DIR, dry_run=dry_run)

    processed = 0
    while True:
        if max_docs is not None and processed >= max_docs:
            break

        job = claim_job(conn, worker_id, dry_run=dry_run)
        if job is None:
            logger.info("[tier1_corpus2events.py] no more pending jobs")
            break

        job_id, corpus, doc_id = job
        started = time.perf_counter()

        try:
            written = process_document(
                conn, embedder, writer, corpus, doc_id, dry_run=dry_run,
            )
            mark_job_done(conn, job_id, dry_run=dry_run)
            elapsed = time.perf_counter() - started
            logger.info(
                "[tier1_corpus2events.py] %s %s/%s  windows=%d  %.1fs",
                "dry-run" if dry_run else "done",
                corpus, doc_id, written, elapsed,
            )
        except Exception as exc:
            logger.exception("[window_embedder] failed %s/%s", corpus, doc_id)
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
        "[tier1_corpus2events.py] worker %s finished (%d documents)",
        worker_id, processed,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Medium-window MacBERTh embedder")
    p.add_argument("--populate", action="store_true", help="Populate embedding_jobs from documents table")
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
        logger.info("[tier1_corpus2events.py] Shall populated jobs")
        n = populate_jobs(
            conn,
            corpus=args.corpus,
            min_year=args.min_year,
            max_year=args.max_year,
        )
        logger.info("[tier1_corpus2events.py] populated %d jobs", n)
        conn.close()
        return

    conn.close()

    run_worker(
        worker_id=args.worker_id,
        max_docs=args.max_docs,
        backend=args.backend,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

