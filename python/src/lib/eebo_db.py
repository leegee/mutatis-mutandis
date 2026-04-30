# lib/eebo_db.py

from __future__ import annotations

import os
import time
import psycopg
from psycopg import sql, Connection
import hashlib

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
                    DROP TABLE IF EXISTS vector_map CASCADE;
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

                CREATE TABLE tokens (
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    token TEXT NOT NULL,
                    raw_token TEXT,
                    canonical TEXT,
                    PRIMARY KEY (doc_id, token_idx),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE TABLE vector_map (
                    vector_id BIGINT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,

                    UNIQUE (doc_id, token_idx),
                    FOREIGN KEY (doc_id, token_idx)
                        REFERENCES tokens(doc_id, token_idx)
                        ON DELETE CASCADE
                );

                CREATE MATERIALIZED VIEW pamphlet_vectors AS
                    SELECT
                        v.vector_id,
                        v.doc_id,
                        v.token_idx,
                        t.token,
                        d.pub_year,
                        d.slice_start,
                        d.slice_end
                    FROM vector_map v
                    JOIN tokens t
                    ON t.doc_id = v.doc_id
                    AND t.token_idx = v.token_idx
                    JOIN documents d
                    ON d.doc_id = v.doc_id;
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


def create_vector_map_indexes(conn: Connection) -> None:
    logger.info("Creating vector_map indexes")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_map_doc_token ON vector_map(doc_id, token_idx);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vector_map_vector_id ON vector_map(vector_id);")


# VECTOR MAP POPULATION
def vector_id(doc_id: str, token_idx: int) -> int:
    h = hashlib.blake2b(f"{doc_id}:{token_idx}".encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")

def build_vector_map(conn):
    logger.info("Building vector_map")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("TRUNCATE vector_map;")

            for doc_id, token_idx, _ in canonical_token_stream(conn):
                cur.execute(
                    """
                    INSERT INTO vector_map (vector_id, doc_id, token_idx)
                    VALUES (%s, %s, %s)
                    """,
                    (vector_id(doc_id, token_idx), doc_id, token_idx),
                )

def populate_vector_map(conn) -> None:
    """
    Rebuilds vector identity from canonical token stream.

    Invariant:
        vector_map is fully deterministic and must be rebuilt from scratch.
    """

    logger.info("Resetting vector_map")

    with conn.transaction():
        with conn.cursor() as cur:

            # CRITICAL: ensure idempotent rebuild
            cur.execute("TRUNCATE vector_map RESTART IDENTITY;")

            logger.info("Rebuilding vector_map from tokens")

            cur.execute("""
                SELECT doc_id, token_idx
                FROM tokens
                ORDER BY doc_id, token_idx
            """)

            for doc_id, token_idx in cur.fetchall():

                vector_id = abs(hash(f"{doc_id}:{token_idx}"))

                cur.execute(
                    """
                    INSERT INTO vector_map (vector_id, doc_id, token_idx)
                    VALUES (%s, %s, %s)
                    """,
                    (vector_id, doc_id, token_idx),
                )

    logger.info("vector_map rebuild complete")


def drop_tokens_fk(conn: Connection) -> None:
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE tokens DROP CONSTRAINT IF EXISTS tokens_doc_id_fkey;")
