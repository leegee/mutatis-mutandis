from datetime import datetime
import json
import sqlite3

from fast_api.connections import get_jobs_conn


def now():
    return datetime.utcnow().isoformat()


def init_db():
    conn = get_jobs_conn()

    conn.execute("DROP TABLE IF EXISTS jobs;")

    conn.execute("""
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        payload TEXT,
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

    conn.execute("""
    CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs(status, created_at)
    """)

    conn.commit()

    # recover interrupted jobs
    conn.execute("""
        UPDATE jobs
        SET status='queued',
            stage=NULL
        WHERE status='running'
    """)

    conn.commit()


def create_job(*, job_id: str, concept: str, job_type: str = "default", payload: dict = None):
    get_jobs_conn().execute("""
        INSERT INTO jobs (
            job_id, concept, status,  created_at, job_type, payload
        )
        VALUES (
            ?,      ?,      'queued', ?,        ?,        ?
        )
    """, (
        job_id, concept,              now(),       job_type,  json.dumps(payload) if payload else None
    ))
    get_jobs_conn().commit()


def claim_next_job():
    sql = """
    UPDATE jobs
    SET status='running',
        started_at=?,
        last_heartbeat=?,
        attempts=attempts+1
    WHERE job_id = (
        SELECT job_id
        FROM jobs
        WHERE status='queued'
        ORDER BY created_at
        LIMIT 1
    )
    AND status='queued'
    RETURNING job_id, concept, job_type;
    """

    conn = get_jobs_conn()
    row = conn.execute(sql, (now(), now())).fetchone()

    if row is None:
        return None

    job_id, concept, job_type = row

    return {
        "job_id": job_id,
        "concept": concept,
        "job_type": job_type,
    }


def heartbeat(job_id: str):
    conn = get_jobs_conn()

    conn.execute("""
        UPDATE jobs
        SET last_heartbeat=?
        WHERE job_id=?
    """, (now(), job_id))

    conn.commit()


def mark_done(job_id: str):
    conn = get_jobs_conn()

    conn.execute("""
        UPDATE jobs
        SET status='done',
            stage='complete',
            finished_at=?
        WHERE job_id=?
    """, (now(), job_id))

    conn.commit()


def mark_error(job_id: str, error: str):
    conn = get_jobs_conn()

    conn.execute("""
        UPDATE jobs
        SET status='error',
            error=?,
            finished_at=?
        WHERE job_id=?
    """, (error, now(), job_id))

    conn.commit()


def update_stage(job_id: str, stage: str):
    conn = get_jobs_conn()

    conn.execute("""
        UPDATE jobs
        SET stage=?,
            last_heartbeat=?
        WHERE job_id=?
    """, (stage, now(), job_id))

    conn.commit()


def get_job(job_id: str):
    return get_jobs_conn().execute(""" SELECT * FROM jobs WHERE job_id=? """, (job_id,)).fetchone()


def list_jobs():
    conn = get_jobs_conn()
    return conn.execute(""" SELECT * FROM jobs ORDER BY created_at DESC """).fetchall()

