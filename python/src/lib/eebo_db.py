# lib/eebo_db.py

"""
eebo_db.py - EEBO database access

Connections, schema (wip that should use .sql files), etc

"""

import os
import psycopg
from psycopg import sql, Connection
import time

from lib.eebo_logging import logger
import lib.eebo_config as config

_DB_RETRIES = 3
_DB_RETRY_DELAY = 5  # seconds
dbname = os.environ.get("PGDATABASE", "eebo")
host = os.environ.get("PGHOST", "localhost")
user = os.environ.get("PGUSER", "postgres")
password = os.environ.get("PGPASSWORD")
port = os.environ.get("PGPORT", 5432)

def get_connection(
    *,
    connect_timeout: int = 5,
    application_name: str = "eebo",
) -> Connection:
    """
    Create and return a PostgreSQL connection with autocommit disabled.
    Callers should use `with conn.transaction():` or call `conn.commit()`.
    Applies session-level tuning suitable for large bulk ingestion.
    """
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

            # Session-level tuning
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET work_mem = '128MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET temp_buffers = '32MB';")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"PostgreSQL connection attempt {attempt}/{_DB_RETRIES} failed: {exc}"
            )
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError("Could not establish PostgreSQL connection") from last_exc


def get_autocommit_connection(
    *,
    connect_timeout: int = 5,
    application_name: str = "eebo",
) -> Connection:
    """
    Get a fresh PostgreSQL connection in autocommit mode.
    Safe for COPY / bulk insert operations.
    Applies session-level tuning suitable for high-speed ingestion.
    """
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
                autocommit=True,  # enable immediately on connect
            )

            # Session-level tuning for bulk insert
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")
                cur.execute("SET work_mem = '128MB';")
                cur.execute("SET maintenance_work_mem = '1GB';")
                cur.execute("SET temp_buffers = '32MB';")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"PostgreSQL autocommit connection attempt {attempt}/{_DB_RETRIES} failed: {exc}"
            )
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError(
        "Could not establish PostgreSQL autocommit connection"
    ) from last_exc



def init_db(conn: Connection, drop_existing: bool = True) -> None:
    """
    Initialise database schema.
    If drop_existing=True, all existing tables are dropped first.
    Intended for clean re-ingestion runs.
    """
    logger.info("Initialising database schema")

    with conn.transaction():
        with conn.cursor() as cur:
            if drop_existing:
                logger.info("Dropping existing tables")
                cur.execute("""
                    DROP MATERIALIZED VIEW IF EXISTS document_search CASCADE;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_tokens CASCADE;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;

                    DROP TABLE IF EXISTS tokens CASCADE;
                    DROP TABLE IF EXISTS documents CASCADE;

                    DROP SEQUENCE IF EXISTS vector_id_seq;
                """)

            logger.info("Creating tables")
            cur.execute("""
                /* Core document metadata */
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY,
                    filepath TEXT,
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
                    raw_token text,
                    canonical TEXT,
                    vector_id BIGINT UNIQUE DEFAULT nextval('vector_id_seq'),
                    PRIMARY KEY (doc_id, token_idx),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
            """)

    logger.info("Database schema created")



def drop_token_indexes(conn: Connection) -> None:
    logger.info("Dropping token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                DROP INDEX IF EXISTS idx_tokens_token;
                DROP INDEX IF EXISTS idx_tokens_doc;
                DROP INDEX IF EXISTS idx_tokens_canonical;
                DROP INDEX IF EXISTS idx_tokens_token_lower;
                DROP INDEX IF EXISTS idx_documents_lang;
                DROP INDEX IF EXISTS idx_documents_filepath;
            """)
    logger.info("Token indexes dropped")


def create_token_indexes(conn: Connection) -> None:
    logger.info("Creating basic token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc ON tokens(doc_id);")
            cur.execute("CREATE INDEX idx_tokens_token_lower ON tokens (lower(token));")
            cur.execute("CREATE INDEX idx_documents_lang ON documents(lang);")
            cur.execute("CREATE INDEX idx_documents_filepath ON documents(filepath);")

    logger.info("Basic token indexes created")



def create_tiered_token_indexes(conn: Connection) -> None:
    logger.info("Creating tiered token indexes")

    earliest, latest = min(s[0] for s in config.SLICES), max(s[1] for s in config.SLICES)

    # Create non-concurrent indexes and materialized views inside a transaction
    with conn.transaction():
        with conn.cursor() as cur:
            # Index for canonical tokens
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_canonical ON tokens(canonical);")

            # Materialized view for pamphlet_corpus
            logger.info("Creating materialised view pamphlet_corpus")
            cur.execute(f"""
                DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;
                CREATE MATERIALIZED VIEW pamphlet_corpus AS
                SELECT *,
                    CASE
                        WHEN token_count <= 15000 THEN 'core'
                        ELSE 'boundary'
                    END AS corpus_zone
                FROM documents
                WHERE token_count BETWEEN 200 AND 20000
                AND pub_year >= {earliest}
                AND pub_year <= {latest}
                AND title !~* '(tragedy|comedy|farce|interlude|play)'
                AND lang = 'eng';
            """)

            # Index for fast joins (non-concurrent)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pamphlet_corpus_docid ON pamphlet_corpus(doc_id);")

            # Materialized view for pamphlet_tokens
            logger.info("Creating materialised view pamphlet_tokens")
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS pamphlet_tokens AS
                SELECT
                    hashtext(t.doc_id || '_' || t.token_idx) AS token_occurrence_id,
                    t.doc_id,
                    t.token_idx,
                    t.vector_id,
                    t.token,
                    t.canonical
                FROM tokens t
                JOIN pamphlet_corpus d ON t.doc_id = d.doc_id;
            """)

            # Materialized view for document_search
            logger.info("Creating materialised view document_search")
            cur.execute("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS document_search AS
                WITH numbered_tokens AS (
                    SELECT
                        t.doc_id,
                        t.token,
                        t.token_idx,
                        (row_number() OVER (PARTITION BY t.doc_id ORDER BY t.token_idx) - 1) / 50000 AS block_idx
                    FROM tokens t
                    JOIN pamphlet_corpus pc ON pc.doc_id = t.doc_id
                ),
                block_text AS (
                    SELECT
                        doc_id,
                        block_idx,
                        string_agg(token, ' ') AS text
                    FROM numbered_tokens
                    GROUP BY doc_id, block_idx
                )
                SELECT
                    d.doc_id,
                    d.title,
                    d.author,
                    d.pub_year,
                    d.pub_place,
                    d.publisher,
                    bt.block_idx,
                    bt.text,
                    to_tsvector('english', bt.text) AS tsv
                FROM pamphlet_corpus d
                JOIN block_text bt ON bt.doc_id = d.doc_id;
            """)

            # GIN index can stay in transaction
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_search_tsv ON document_search USING GIN(tsv);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_search_docid ON document_search(doc_id);")

    # Create CONCURRENT indexes outside transaction
    logger.info("Creating CONCURRENT indexes")
    # autocommit must be True to allow CONCURRENTLY
    autocommit = conn.autocommit
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_token_occurrence_id ON pamphlet_tokens(token_occurrence_id);")
            cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pamphlet_tokens_docid_slice ON pamphlet_tokens(doc_id, slice_start);")
            cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pt_doc_token_idx ON pamphlet_tokens(doc_id, token, token_idx);")
    finally:
        conn.autocommit = autocommit

    logger.info("Tiered token indexes created")



def drop_tokens_fk(conn: Connection) -> None:
    logger.info("Dropping tokens.doc_id foreign key")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE tokens DROP CONSTRAINT IF EXISTS tokens_doc_id_fkey;")
    logger.info("tokens.doc_id foreign key dropped")


def create_tokens_fk(conn: Connection) -> None:
    logger.info("NOT creating tokens.doc_id foreign key as v slow and immutable data  makes it irrelevant")
    # logger.info("Creating tokens.doc_id foreign key")
    # with conn.transaction():
    #     with conn.cursor() as cur:
    #         cur.execute("""
    #             ALTER TABLE tokens
    #             ADD CONSTRAINT tokens_doc_id_fkey FOREIGN KEY (doc_id)
    #             REFERENCES documents(doc_id)
    #             ON DELETE CASCADE;
    #         """)
    # logger.info("tokens.doc_id foreign key created")


def refresh_views(conn: Connection) -> None:
    logger.info("Refreshing materialized views")

    with conn.transaction():
        with conn.cursor() as cur:
            for view in ["pamphlet_tokens", "pamphlet_corpus", "document_search"]:
                logger.info(f"Refreshing {view}")
                cur.execute(
                    sql.SQL("REFRESH MATERIALIZED VIEW {view}").format( view=sql.Identifier(view) )
                )
    logger.info("All views refreshed")
