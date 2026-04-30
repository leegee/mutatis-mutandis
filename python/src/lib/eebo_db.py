# lib/eebo_db.py

from __future__ import annotations

import os
import time
import psycopg
from psycopg import sql, Connection

from lib.eebo_logging import logger
import lib.eebo_config as config


_DB_RETRIES = 3
_DB_RETRY_DELAY = 5

dbname = os.environ.get("PGDATABASE", "eebo")
host = os.environ.get("PGHOST", "localhost")
user = os.environ.get("PGUSER", "postgres")
password = os.environ.get("PGPASSWORD")
port = os.environ.get("PGPORT", 5432)


def get_connection(*, connect_timeout: int = 5, application_name: str = "eebo") -> Connection:
    last_exc: Exception | None = None

    for attempt in range(1, _DB_RETRIES + 1):
        try:
            conn = psycopg.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port,
                sslmode="disable",
                connect_timeout=connect_timeout,
                application_name=application_name,
            )
            conn.autocommit = False

            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET work_mem = '128MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET temp_buffers = '32MB';")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(f"DB connection attempt {attempt} failed: {exc}")
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError("Could not establish PostgreSQL connection") from last_exc


def get_autocommit_connection(*, connect_timeout: int = 5, application_name: str = "eebo") -> Connection:
    conn = psycopg.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port,
        sslmode="disable",
        connect_timeout=connect_timeout,
        application_name=application_name,
        autocommit=True,
    )

    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = OFF;")
        cur.execute("SET work_mem = '128MB';")
        cur.execute("SET maintenance_work_mem = '1GB';")
        cur.execute("SET temp_buffers = '32MB';")

    return conn


# SCHEMA

def init_db(conn: Connection, drop_existing: bool = True) -> None:
    logger.info("Initialising schema")

    with conn.transaction():
        with conn.cursor() as cur:

            if drop_existing:
                cur.execute("""
                    DROP TABLE IF EXISTS documents CASCADE;
                    DROP TABLE IF EXISTS tokens CASCADE;
                """)

            cur.execute("""
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    pub_year INTEGER,
                    publisher TEXT,
                    pub_place TEXT,
                    source_date_raw TEXT,
                    token_count INTEGER,
                    slice_start INTEGER,
                    slice_end INTEGER,
                    lang CHAR(3) NOT NULL DEFAULT 'eng'
                );

                CREATE SEQUENCE vector_id_seq;

                CREATE TABLE tokens (
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    token TEXT NOT NULL,
                    raw_token TEXT,
                    canonical TEXT,
                    vector_id BIGINT UNIQUE,
                    PRIMARY KEY (doc_id, token_idx),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
            """)

    logger.info("Schema created")


# INDEXES

def create_token_indexes(conn: Connection) -> None:
    logger.info("Creating token indexes")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc ON tokens(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_lower ON tokens(lower(token));")


def drop_tokens_fk(conn: Connection) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE tokens DROP CONSTRAINT IF EXISTS tokens_doc_id_fkey;")
