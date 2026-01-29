# lib/eebo_db.py
import psycopg
from psycopg import Connection
import time

from lib.eebo_logging import logger

_DB_RETRIES = 3
_DB_RETRY_DELAY = 5  # seconds


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
                dbname="eebo",
                user="postgres",
                host="localhost",
                port=5432,
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
                dbname="eebo",
                user="postgres",
                host="localhost",
                port=5432,
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
                    DROP TABLE IF EXISTS documents CASCADE;
                    DROP TABLE IF EXISTS tokens CASCADE;
                    DROP TABLE IF EXISTS token_vectors CASCADE;
                    DROP TABLE IF EXISTS concept_slice_stats;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_corpus CASCADE;
                    DROP INDEX IF EXISTS idx_pamphlet_corpus_docid;
                    DROP MATERIALIZED VIEW IF EXISTS pamphlet_tokens CASCADE;
                    DROP INDEX IF EXISTS idx_pamphlet_tokens_docid_slice;
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
                    token_count INTEGER,
                    slice_start INTEGER,
                    slice_end INTEGER
                );

                CREATE TABLE tokens (
                    doc_id TEXT NOT NULL,
                    token_idx INTEGER NOT NULL,
                    token TEXT NOT NULL,
                    raw_token text,
                    sentence_id INTEGER,
                    canonical TEXT,
                    PRIMARY KEY (doc_id, token_idx),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );

                CREATE TABLE token_vectors (
                    token TEXT NOT NULL,
                    slice_start INT NOT NULL,
                    slice_end   INT NOT NULL,
                    vector FLOAT4[] NOT NULL,
                    PRIMARY KEY (token, slice_start, slice_end)
                );

                CREATE TABLE concept_slice_stats (
                    concept_name TEXT,
                    slice_start  INT,
                    slice_end    INT,
                    centroid     FLOAT4[] NOT NULL,
                    variance     FLOAT4,        -- mean squared distance from centroid
                    token_count  INT,
                    forms_used   TEXT[],        -- which variants actually present
                    PRIMARY KEY (concept_name, slice_start, slice_end)
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
                DROP INDEX IF EXISTS idx_tokens_sentence;
                DROP INDEX IF EXISTS idx_tokens_canonical;
                DROP INDEX IF EXISTS idx_tokens_sentence_notnull;
            """)
    logger.info("Token indexes dropped")


def create_token_indexes(conn: Connection) -> None:
    logger.info("Creating basic token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_token ON tokens(token);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_doc ON tokens(doc_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_sentence ON tokens(sentence_id);")
    logger.info("Basic token indexes created")


def create_tiered_token_indexes(conn: Connection) -> None:
    logger.info("Creating tiered token indexes")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tokens_canonical ON tokens(canonical);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_tokens_sentence_notnull
                ON tokens(sentence_id)
                WHERE sentence_id IS NOT NULL;
            """)

            logger.info("Creating materialised view")

            cur.execute("""
                CREATE MATERIALIZED VIEW document_search AS
                WITH numbered_tokens AS (
                    SELECT
                        doc_id,
                        token,
                        (row_number() OVER (PARTITION BY doc_id ORDER BY token_idx) - 1) / 50000 AS block_idx
                    FROM tokens
                )
                SELECT
                    d.doc_id,
                    d.title,
                    d.author,
                    d.pub_year,
                    d.pub_place,
                    d.publisher,
                    nt.block_idx,
                    to_tsvector('english', string_agg(nt.token, ' ')) AS tsv
                FROM documents d
                JOIN numbered_tokens nt ON nt.doc_id = d.doc_id
                GROUP BY d.doc_id, nt.block_idx, d.title, d.author, d.pub_year, d.pub_place, d.publisher;
            """)
            logger.info("Created materialised view with block-level tsvectors")

            logger.info("Creating GIN index on materialised view")
            cur.execute("CREATE INDEX idx_document_search_tsv ON document_search USING GIN(tsv);")
            logger.info("GIN index created")

            cur.execute("""
                -- Pamphlet-only document materialized view
                CREATE MATERIALIZED VIEW IF NOT EXISTS pamphlet_corpus AS
                SELECT *,
                    CASE
                        WHEN token_count <= 15000 THEN 'core'
                        ELSE 'boundary'
                    END AS corpus_zone
                FROM documents
                WHERE token_count BETWEEN 500 AND 20000
                AND title !~* '(tragedy|comedy|farce|interlude|play)';

                -- Index for fast joins
                CREATE INDEX IF NOT EXISTS idx_pamphlet_corpus_docid
                ON pamphlet_corpus(doc_id);

                -- Refresh when ingesting new:
                -- REFRESH MATERIALIZED VIEW pamphlet_corpus;

                -- Slice-level tokens restricted to pamphlets
                -- Slice-level tokens restricted to pamphlets
                CREATE MATERIALIZED VIEW IF NOT EXISTS pamphlet_tokens AS
                SELECT t.doc_id,
                    t.token_idx,
                    t.token,
                    t.canonical,
                    t.sentence_id,
                    d.corpus_zone,
                    d.pub_year,
                    d.title,
                    d.token_count,
                    d.slice_start,
                    d.slice_end
                FROM tokens t
                JOIN pamphlet_corpus d
                ON t.doc_id = d.doc_id;

                -- Index for performance on slice queries
                CREATE INDEX IF NOT EXISTS idx_pamphlet_tokens_docid_slice
                ON pamphlet_tokens(doc_id, slice_idx);

                -- Refresh when new data ingested:
                -- REFRESH MATERIALIZED VIEW pamphlet_tokens;
            """)

    logger.info("Tiered token indexes created")


def drop_tokens_fk(conn: Connection) -> None:
    logger.info("Dropping tokens.doc_id foreign key")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE tokens DROP CONSTRAINT IF EXISTS tokens_doc_id_fkey;")
    logger.info("tokens.doc_id foreign key dropped")


def create_tokens_fk(conn: Connection) -> None:
    logger.info("Creating tokens.doc_id foreign key")
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE tokens
                ADD CONSTRAINT tokens_doc_id_fkey FOREIGN KEY (doc_id)
                REFERENCES documents(doc_id)
                ON DELETE CASCADE;
            """)

    # UPDATE documents d
    # SET tsv = to_tsvector('english', (SELECT string_agg(token, ' ') FROM tokens t WHERE t.doc_id = d.doc_id))
    # WHERE d.doc_id = 'A00001';

    logger.info("tokens.doc_id foreign key created")

