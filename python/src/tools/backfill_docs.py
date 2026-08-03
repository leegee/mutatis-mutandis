#!/usr/bin/env python3
"""
Backfill Tier 2 SQLite document metadata from PostgreSQL.

Only documents already referenced by Tier 2 are touched.
The EEBO catalogue is never loaded into memory.
"""

from __future__ import annotations

import json
import sqlite3

from lib.eebo_config import CORPUS_TIER2_DB_PATH
from lib.eebo_db import get_connection


BATCH_SIZE = 1000


def sqlite_connection(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_missing_doc_ids(sqlite_con):
    rows = sqlite_con.execute(
        """
        SELECT doc_id
        FROM documents
        WHERE title IS NULL
           OR author IS NULL
           OR pub_year IS NULL
        """
    )

    return [
        row[0]
        for row in rows
    ]


def fetch_pg_documents(pg_con, doc_ids):
    placeholders = ",".join( ["%s"] * len(doc_ids) )

    sql = f"""
        SELECT
            doc_id,
            title,
            author,
            pub_year,
            publisher,
            pub_place
        FROM documents
        WHERE doc_id IN ({placeholders})
    """

    with pg_con.cursor() as cur:
        cur.execute(
            sql,
            doc_ids,
        )
        return cur.fetchall()


def update_sqlite(sqlite_con, rows):
    sqlite_con.executemany(
        """
        UPDATE documents
        SET
            title = ?,
            author = ?,
            pub_year = ?,
            publisher = ?,
            pub_place = ?
        WHERE doc_id = ?
        """,
        [
            (
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                str(row[0]),
            )
            for row in rows
        ],
    )
    sqlite_con.commit()


def main():
    sqlite_con = sqlite_connection(
        CORPUS_TIER2_DB_PATH
    )

    pg_con = get_connection()

    try:

        doc_ids = fetch_missing_doc_ids(
            sqlite_con
        )

        print(
            f"Need to enrich {len(doc_ids):,} documents"
        )

        updated = 0

        for batch in chunks(
            doc_ids,
            BATCH_SIZE,
        ):

            rows = fetch_pg_documents(
                pg_con,
                batch,
            )

            if rows:
                update_sqlite(
                    sqlite_con,
                    rows,
                )

                updated += len(rows)

            print(
                f"Updated {updated:,}/{len(doc_ids):,}"
            )

        print(
            "Done"
        )

    finally:
        sqlite_con.close()
        pg_con.close()


if __name__ == "__main__":
    main()
