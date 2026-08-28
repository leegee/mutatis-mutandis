"""
tier2/populate_sqlite_documents.py

Populate Tier 2 SQLite document metadata from the corpus PostgreSQL database.

Only documents referenced by Tier 2 data are copied. The corpus PostgreSQL
database remains the source of truth for document metadata.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_db import get_connection, analysis_db_connection
from lib.corpus_logging import logger


SQLITE_DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id         TEXT PRIMARY KEY,
    corpus         TEXT,
    filepath       TEXT,
    title          TEXT,
    author         TEXT,
    pub_year       INTEGER,
    publisher      TEXT,
    pub_place      TEXT,
    source_date_raw TEXT,
    token_count    INTEGER,
    lang           TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_pub_year
    ON documents(pub_year);

CREATE INDEX IF NOT EXISTS idx_documents_author
    ON documents(author);
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the Tier 2 SQLite documents table from the corpus PostgreSQL documents table."
        ),
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the SQLite documents table before importing.",
    )

    return parser.parse_args()


def referenced_doc_ids(
    conn: sqlite3.Connection,
) -> list[str]:
    """
    Return all document IDs referenced anywhere in the Tier 2 database.

    Events are currently sufficient because every neighbour is associated
    with an event, but neighbours are included explicitly so this remains
    correct if the schema later permits neighbour-only documents.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT doc_id
        FROM events
        WHERE doc_id IS NOT NULL

        UNION

        SELECT DISTINCT doc_id
        FROM neighbours
        WHERE doc_id IS NOT NULL;
        """
    ).fetchall()

    return [
        row[0]
        for row in rows
    ]


def fetch_documents(
    doc_ids: list[str],
) -> list[tuple]:
    """
    Fetch document metadata from PostgreSQL.

    The PostgreSQL documents table remains authoritative. Missing IDs are
    tolerated here and reported by the caller rather than causing unrelated
    Tier 2 results to be discarded.
    """
    if not doc_ids:
        return []

    started = time.perf_counter()

    with get_connection(
        application_name="tier2-populate-documents",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    doc_id,
                    corpus,
                    author,
                    title,
                    pub_year,
                    pub_place
                FROM documents
                WHERE doc_id = ANY(%s)
                """,
                (doc_ids,),
            )

            rows = cur.fetchall()

    logger.info(
        "[tier2 documents] fetched %d PostgreSQL documents in %.3fs",
        len(rows),
        time.perf_counter() - started,
    )

    return rows


def populate_documents(
    sqlite_path: Path,
    *,
    clear: bool = False,
) -> None:
    started = time.perf_counter()

    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"Tier 2 SQLite database does not exist: {sqlite_path}"
        )

    with analysis_db_connection(sqlite_path) as sqlite_conn:
        sqlite_conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        sqlite_conn.executescript(
            SQLITE_DOCUMENTS_SCHEMA
        )

        if clear:
            logger.info(
                "[tier2 documents] clearing existing documents table"
            )

            sqlite_conn.execute(
                "DELETE FROM documents"
            )

        doc_ids = referenced_doc_ids(
            sqlite_conn
        )

        logger.info(
            "[tier2 documents] found %d referenced document IDs",
            len(doc_ids),
        )

        if not doc_ids:
            logger.warning(
                "[tier2 documents] no referenced documents found"
            )
            return

        rows = fetch_documents(
            doc_ids
        )

        found_ids = {
            row[0]
            for row in rows
        }

        missing_ids = (
            set(doc_ids)
            - found_ids
        )

        if missing_ids:
            logger.warning(
                "[tier2 documents] %d referenced document IDs "
                "were not found in PostgreSQL",
                len(missing_ids),
            )

        sqlite_conn.executemany(
            """
            INSERT INTO documents (
                doc_id,
                corpus,
                author,
                title,
                pub_year,
                pub_place
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                corpus = excluded.corpus,
                author = excluded.author,
                title = excluded.title,
                pub_year = excluded.pub_year,
                pub_place = excluded.pub_place,
            """,
            rows,
        )

        sqlite_conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
                idx_documents_pub_year
            ON documents(pub_year)
            """
        )

        sqlite_conn.commit()

    logger.info(
        "[tier2 documents] populated %d documents in %.3fs",
        len(rows),
        time.perf_counter() - started,
    )


def main() -> None:
    args = parse_args()

    populate_documents(
        CORPUS_TIER2_DB_PATH,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()

