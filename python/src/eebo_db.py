import sys
import psycopg
from eebo_logging import logger

def get_connection():
    """
    Connect using libpq environment variables:
    PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT
    """
    try:
        return psycopg.connect()
    except Exception as exc:
        print(f"[ERROR] Cannot open PostgreSQL database: {exc}")
        sys.exit(1)


dbh = get_connection()


def init_db(drop_existing=True):
    """
    Initialise database schema.

    If drop_existing=True, all existing tables are dropped first.
    Intended for clean re-ingestion runs.
    """
    with dbh.cursor() as cur:

        if drop_existing:
            logger.info("Dropping all tables")
            cur.execute("""
                DROP TABLE IF EXISTS neighbourhoods CASCADE;
                DROP TABLE IF EXISTS sentences CASCADE;
                DROP TABLE IF EXISTS tokens CASCADE;
                DROP TABLE IF EXISTS spelling_map CASCADE;
                DROP TABLE IF EXISTS documents CASCADE;
                DROP TABLE IF EXISTS ingest_runs CASCADE;
            """)
            logger.info("Dropped all tables")

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

        logger.info("Created tables")


def drop_token_indexes():
    logger.info("Dropping table indexes")
    with dbh.cursor() as cur:
        cur.execute("""
            DROP INDEX IF EXISTS idx_tokens_token;
            DROP INDEX IF EXISTS idx_tokens_doc;
            DROP INDEX IF EXISTS idx_tokens_sentence;
        """)
    dbh.commit()
    logger.info("Dropped table indexes")


def create_token_indexes():
    logger.info("Creating table indexes")
    with dbh.cursor() as cur:
        cur.execute("""
            CREATE INDEX idx_tokens_token ON tokens(token);
            CREATE INDEX idx_tokens_doc ON tokens(doc_id);
            CREATE INDEX idx_tokens_sentence ON tokens(sentence_id);
        """)
    dbh.commit()
    logger.info("Created table indexes")
