import sqlite3
from pathlib import Path
from datetime import datetime
from lib.eebo_config import OUT_DIR

DB_PATH = OUT_DIR / "fastapi_jobs.sqlite3"

def now():
    return datetime.utcnow().isoformat()


def get_conn():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    con.execute("PRAGMA busy_timeout=5000;")
    return con


def init_db():
    con = get_conn()

    con.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        concept TEXT NOT NULL,
        status TEXT NOT NULL,
        stage TEXT,
        attempts INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        last_heartbeat TEXT,
        error TEXT
    )
    """)

    con.execute("""
    CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)
    """)

    con.commit()

    # recover interrupted jobs
    con.execute("""
        UPDATE jobs
        SET status='queued',
            stage=NULL
        WHERE status='running'
    """)

    con.commit()
    con.close()


def create_job(job_id: str, concept: str):
    con = get_conn()

    con.execute("""
        INSERT INTO jobs (
            job_id, concept, status, created_at
        )
        VALUES (
            ?, ?, 'queued', ?
        )
    """, (
        job_id,
        concept,
        now(),
    ))

    con.commit()
    con.close()


def claim_next_job():
    con = get_conn()

    try:
        con.execute("BEGIN IMMEDIATE")

        row = con.execute("""
            SELECT job_id, concept
            FROM jobs
            WHERE status='queued'
            ORDER BY created_at
            LIMIT 1
        """).fetchone()

        if row is None:
            con.commit()
            return None

        job_id, concept = row

        con.execute("""
            UPDATE jobs
            SET status='running',
                started_at=?,
                last_heartbeat=?,
                attempts=attempts+1
            WHERE job_id=?
        """, (
            now(),
            now(),
            job_id,
        ))

        con.commit()

        return {
            "job_id": job_id,
            "concept": concept,
        }

    finally:
        con.close()


def heartbeat(job_id):
    con = get_conn()

    con.execute("""
        UPDATE jobs
        SET last_heartbeat=?
        WHERE job_id=?
    """, (
        now(),
        job_id,
    ))

    con.commit()
    con.close()


def mark_done(job_id):
    con = get_conn()

    con.execute("""
        UPDATE jobs
        SET status='done',
            stage='complete',
            finished_at=?
        WHERE job_id=?
    """, (
        now(),
        job_id,
    ))

    con.commit()
    con.close()


def mark_error(job_id, error):
    con = get_conn()

    con.execute("""
        UPDATE jobs
        SET status='error',
            error=?,
            finished_at=?
        WHERE job_id=?
    """, (
        error,
        now(),
        job_id,
    ))

    con.commit()
    con.close()


def update_stage(job_id, stage):
    con = get_conn()

    con.execute("""
        UPDATE jobs
        SET stage=?,
            last_heartbeat=?
        WHERE job_id=?
    """, (
        stage,
        now(),
        job_id,
    ))

    con.commit()
    con.close()


def get_job(job_id):
    con = get_conn()

    row = con.execute("""
        SELECT * FROM jobs WHERE job_id=?
    """, (job_id,)).fetchone()

    con.close()
    return row
