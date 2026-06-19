# fast_api/jobs_dao.py
from pathlib import Path
from datetime import datetime

import sqlite3

from lib.eebo_config import OUT_DIR
from lib.eebo_logging import logger

JOBS_DB_PATH = OUT_DIR / "fastapi_jobs.sqlite3"

def now():
    return datetime.utcnow().isoformat()


def get_jobs_conn():
    jobs_con = sqlite3.connect(JOBS_DB_PATH)
    jobs_con.execute("PRAGMA journal_mode=WAL;")
    jobs_con.execute("PRAGMA synchronous=NORMAL;")
    jobs_con.execute("PRAGMA busy_timeout=5000;")
    return jobs_con


def init_db():
    jobs_con = get_jobs_conn()
    jobs_con.execute("""
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

    jobs_con.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at)
    """)
    jobs_con.commit()

    # recover interrupted jobs
    jobs_con.execute("""
        UPDATE jobs
        SET status='queued',
            stage=NULL
        WHERE status='running'
    """)

    jobs_con.commit()


def create_job(job_id: str, concept: str):
    jobs_con = get_jobs_conn()
    jobs_con.execute("""
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

    jobs_con.commit()


def claim_next_job():
    # logger.debug("[fast_api.jobs_dao.claim_next_job Enter]");
    jobs_con = get_jobs_conn()

    try:
        jobs_con.execute("BEGIN IMMEDIATE")
        row = jobs_con.execute("""
            SELECT job_id, concept
            FROM jobs
            WHERE status='queued'
            ORDER BY created_at
            LIMIT 1
        """).fetchone()

        # No jobs queued? Leave.
        if row is None:
            return None

        job_id, concept = row

        # Take the job: updated job status to indicate it is not queued but running
        jobs_con.execute("""
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

        jobs_con.commit()

        return {
            "job_id": job_id,
            "concept": concept,
        }

    finally:
        logger.deubg("[fast_api.jobs_dao.claim_next_job Complete]");


def heartbeat(job_id):
    jobs_con = get_jobs_conn()

    con.execute("""UPDATE jobs SET last_heartbeat = ? WHERE job_id = ?""", (
        now(),
        job_id,
    ))

    row = jobs_con.execute(""" SELECT job_id, last_heartbeat FROM jobs LIMIT 1 """).fetchone()

    # Odd failure
    if row is None:
        raise ValueError(f"Could not find the job I thought I'd just created, {job_id}")
        return None

    job_id, concept = row

    con.commit()
    return {
            "job_id": job_id,
            "last_heartbeat": last_heartbeat,
        }


def mark_done(job_id):
    jobs_conn = get_jobs_conn()

    jobs_con.execute("""
        UPDATE jobs
        SET status='done',
            stage='complete',
            finished_at=?
        WHERE job_id=?
    """, (
        now(),
        job_id,
    ))

    jobs_con.commit()


def mark_error(job_id, error):
    jobs_con = get_jobs_conn()

    jobs_con.execute("""
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

    jobs_con.commit()


def update_stage(job_id, stage):
    jobs_con = get_jobs_conn()

    jobs_con.execute("""
        UPDATE jobs
        SET stage=?,
            last_heartbeat=?
        WHERE job_id=?
    """, (
        stage,
        now(),
        job_id,
    ))

    jobs_con.commit()


def get_job(job_id):
    jobs_con = get_jobs_conn()

    row = jobs_con.execute("""
        SELECT * FROM jobs WHERE job_id=?
    """, (job_id,)).fetchone()

    jobs_
    return row

def list_jobs():

    jobs_con = get_jobs_connection()

    rows = jobs_con.execute("""
        SELECT *
        FROM jobs
        ORDER BY created_at DESC
    """).fetchall()

    return rows
