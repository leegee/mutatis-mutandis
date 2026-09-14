#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_db import analysis_db_connection


_SCHEMA_INIT = """
CREATE TABLE IF NOT EXISTS event_field (
    concept   TEXT    NOT NULL,
    event_id  INTEGER NOT NULL,
    role      TEXT    NOT NULL,

    PRIMARY KEY (concept, event_id),

    FOREIGN KEY (concept)
        REFERENCES concepts(concept),

    CHECK (role IN ('seed', 'neighbour', 'both'))
);

CREATE INDEX IF NOT EXISTS idx_event_field_concept
    ON event_field(concept);

CREATE INDEX IF NOT EXISTS idx_event_field_event
    ON event_field(event_id);
"""


def rebuild_event_field(con, concept: str) -> None:
    con.execute(
        """
        DELETE FROM event_field
        WHERE concept = ?
        """,
        (concept,),
    )

    con.execute(
        """
        INSERT INTO event_field (
            concept,
            event_id,
            role
        )
        SELECT
            ?,
            s.event_id,
            CASE
                WHEN EXISTS (
                    SELECT 1
                    FROM neighbour_edges ne
                    JOIN retrieval_runs rr
                      ON rr.run_id = ne.run_id
                    WHERE rr.concept = ?
                      AND ne.neighbour_event_id = s.event_id
                )
                THEN 'both'
                ELSE 'seed'
            END
        FROM concept_seeds s
        WHERE s.concept = ?

        UNION

        SELECT
            ?,
            ne.neighbour_event_id,
            'neighbour'
        FROM neighbour_edges ne
        JOIN retrieval_runs rr
          ON rr.run_id = ne.run_id
        WHERE rr.concept = ?
          AND NOT EXISTS (
              SELECT 1
              FROM concept_seeds s
              WHERE s.concept = ?
                AND s.event_id = ne.neighbour_event_id
          )
        """,
        (
            concept,
            concept,
            concept,
            concept,
            concept,
            concept,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and populate Tier 2 event_field."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
        help="Existing Tier 2 SQLite database",
    )
    parser.add_argument(
        "--concept",
        required=True,
        help="Concept to rebuild",
    )

    args = parser.parse_args()
    concept = args.concept.upper()

    con = analysis_db_connection(args.db)

    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(_SCHEMA_INIT)

        exists = con.execute(
            """
            SELECT 1
            FROM concepts
            WHERE concept = ?
            """,
            (concept,),
        ).fetchone()

        if exists is None:
            raise RuntimeError(
                f"Concept {concept!r} does not exist in {args.db}"
            )

        rebuild_event_field(con, concept)

        con.commit()

        total = con.execute(
            """
            SELECT COUNT(*)
            FROM event_field
            WHERE concept = ?
            """,
            (concept,),
        ).fetchone()[0]

        roles = con.execute(
            """
            SELECT role, COUNT(*)
            FROM event_field
            WHERE concept = ?
            GROUP BY role
            ORDER BY role
            """,
            (concept,),
        ).fetchall()

        print(f"Database: {args.db}")
        print(f"Concept:  {concept}")
        print(f"Events:   {total:,}")

        for role, count in roles:
            print(f"  {role}: {count:,}")

    except Exception:
        con.rollback()
        raise

    finally:
        con.close()


if __name__ == "__main__":
    main()