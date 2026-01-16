# eebo_db.py
import psycopg
from psycopg import Connection
import time

from eebo_logging import logger

_DB_RETRIES = 3
_DB_RETRY_DELAY = 5  # seconds

def get_connection(
    *,
    connect_timeout: int = 5,
    application_name: str = "eebo",
) -> Connection:
    """
    Create and return a PostgreSQL connection.

    Autocommit is disabled; callers are expected to use
    `with conn.transaction():` or call `conn.commit()`.
    """
    last_exc: Exception | None = None

    for attempt in range(1, _DB_RETRIES + 1):
        try:
            conn = psycopg.connect(
                dbname="eebo",
                user="postgres",
                host="localhost",
                port=5432,
                connect_timeout=connect_timeout,
                application_name=application_name,
            )
            conn.autocommit = False

            # Ingest / bulk-read optimised settings
            with conn.cursor() as cur:
                cur.execute("SET synchronous_commit = OFF;")

            return conn

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"PostgreSQL connection attempt {attempt}/{_DB_RETRIES} failed: {exc}"
            )
            if attempt < _DB_RETRIES:
                time.sleep(_DB_RETRY_DELAY)

    raise RuntimeError("Could not establish PostgreSQL connection") from last_exc


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
                    DROP TABLE IF EXISTS neighbourhoods CASCADE;
                    DROP TABLE IF EXISTS sentences CASCADE;
                    DROP TABLE IF EXISTS tokens CASCADE;
                    DROP TABLE IF EXISTS spelling_map CASCADE;
                    DROP TABLE IF EXISTS documents CASCADE;
                    DROP TABLE IF EXISTS ingest_runs CASCADE;
                """)

            logger.info("Creating tables")
            cur.execute("""
                /* Core document metadata */
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT,
                    author TEXT,
                    pub_year INTEGER,
                    publisher TEXT,
                    pub_place TEXT,
                    source_date_raw TEXT,

                    -- analysis helpers
                    token_count INTEGER,
                    slice_start INTEGER,
                    slice_end INTEGER
                );

                /* Token index (authoritative) */
                CREATE TABLE tokens (
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    token TEXT NOT NULL,

                    -- future-proofing (not populated at ingest)
                    sentence_id INTEGER,
                    canonical TEXT,

                    PRIMARY KEY (doc_id, token_idx),
                    FOREIGN KEY (doc_id)
                        REFERENCES documents(doc_id)
                        ON DELETE CASCADE
                );

                /* Sentence table (derived, optional, post-ingest) */
                CREATE TABLE sentences (
                    doc_id TEXT NOT NULL,
                    sentence_id INTEGER NOT NULL,
                    sentence_text_raw TEXT,
                    sentence_text_norm TEXT,
                    embedding DOUBLE PRECISION[],

                    PRIMARY KEY (doc_id, sentence_id),
                    FOREIGN KEY (doc_id)
                        REFERENCES documents(doc_id)
                        ON DELETE CASCADE
                );

                /* Orthography / variant control */
                CREATE TABLE spelling_map (
                    variant TEXT PRIMARY KEY,
                    canonical TEXT NOT NULL,
                    concept_type TEXT NOT NULL DEFAULT 'orthographic',
                    CHECK (
                        concept_type IN ('orthographic','derivational','exclude')
                    )
                );

                /* Diachronic neighbourhood output */
                CREATE TABLE neighbourhoods (
                    slice_start INTEGER NOT NULL,
                    slice_end INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    neighbour TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    cosine DOUBLE PRECISION,

                    PRIMARY KEY (slice_start, slice_end, query, rank)
                );

                /* Optional provenance tracking */
                CREATE TABLE ingest_runs (
                    run_id SERIAL PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT now(),
                    code_version TEXT,
                    notes TEXT
                );
            """)

    logger.info("Database schema created")


# ---------------------------------------------------------------------
# Index management (explicit, post-ingest)
# ---------------------------------------------------------------------

def drop_token_indexes(conn: Connection) -> None:
    """
    Drop all token-related indexes before bulk ingestion.
    """
    logger.info("Dropping token indexes")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                DROP INDEX IF EXISTS idx_tokens_token;
                DROP INDEX IF EXISTS idx_tokens_doc;
                DROP INDEX IF EXISTS idx_tokens_sentence;
                DROP INDEX IF EXISTS idx_tokens_canonical;
                DROP INDEX IF EXISTS idx_tokens_sentence_notnull;
            """)

    logger.info("Token indexes dropped")


def create_token_indexes(conn: Connection) -> None:
    """
    Create basic token indexes for post-ingest querying.
    """
    logger.info("Creating basic token indexes")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_token
                    ON tokens(token);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_doc
                    ON tokens(doc_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_sentence
                    ON tokens(sentence_id);
            """)

    logger.info("Basic token indexes created")


def create_tiered_token_indexes(conn: Connection) -> None:
    """
    Create additional / tiered indexes for canonicalisation
    and sentence-level queries. Run only after ingestion.
    """
    logger.info("Creating tiered token indexes")

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_canonical
                    ON tokens(canonical);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_sentence_notnull
                    ON tokens(sentence_id)
                    WHERE sentence_id IS NOT NULL;
            """)

    logger.info("Tiered token indexes created")
