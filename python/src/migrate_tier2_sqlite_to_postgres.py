from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from psycopg import Connection

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_db import get_connection, create_tier2_schema
from lib.corpus_logging import logger

BATCH_SIZE = 10_000


def sqlite_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def postgres_count(
    conn: Connection,
    table: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        row = cur.fetchone()
        assert row is not None
        return int(row[0])


def sqlite_event_ids(
    conn: sqlite3.Connection,
    query: str,
    params: tuple = (),
) -> set[int]:
    return {
        int(row[0])
        for row in conn.execute(query, params)
    }


def validate_event_ids(
    pg: Connection,
    event_ids: set[int],
    label: str,
) -> None:
    if not event_ids:
        return

    missing: set[int] = set()

    ids = list(event_ids)

    with pg.cursor() as cur:
        for start in range(0, len(ids), BATCH_SIZE):
            batch = ids[start:start + BATCH_SIZE]

            cur.execute(
                """
                SELECT event_id
                FROM events
                WHERE event_id = ANY(%s)
                """,
                (batch,),
            )

            found = {int(row[0]) for row in cur.fetchall()}
            missing.update(set(batch) - found)

    if missing:
        sample = sorted(missing)[:20]
        raise RuntimeError(
            f"{label}: {len(missing)} event IDs are missing from "
            f"PostgreSQL events; sample={sample}"
        )


def validate_source_events(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    logger.info("[tier2-migrate] validating event references")

    concept_seed_ids = sqlite_event_ids(
        sqlite,
        "SELECT DISTINCT event_id FROM concept_seeds",
    )
    validate_event_ids(
        pg,
        concept_seed_ids,
        "concept_seeds",
    )

    field_ids = sqlite_event_ids(
        sqlite,
        "SELECT DISTINCT event_id FROM event_field",
    )
    validate_event_ids(
        pg,
        field_ids,
        "event_field",
    )

    edge_ids = sqlite_event_ids(
        sqlite,
        """
        SELECT seed_event_id FROM neighbour_edges
        UNION
        SELECT neighbour_event_id FROM neighbour_edges
        UNION
        SELECT via_event_id
        FROM neighbour_edges
        WHERE via_event_id IS NOT NULL
        """,
    )
    validate_event_ids(
        pg,
        edge_ids,
        "neighbour_edges",
    )

    logger.info("[tier2-migrate] event reference validation passed")


def migrate_concepts(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT concept, n_events
        FROM concepts
        ORDER BY concept
    """).fetchall()

    with pg.cursor() as cur:
        for concept, n_events in rows:
            cur.execute(
                """
                INSERT INTO tier2.concepts (
                    concept,
                    n_events
                )
                VALUES (%s, %s)
                ON CONFLICT (concept) DO UPDATE
                SET n_events = EXCLUDED.n_events
                """,
                (concept, int(n_events)),
            )


def migrate_retrieval_runs(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            run_id,
            concept,
            from_year,
            to_year,
            seed_population,
            neighbour_population,
            scales,
            top_n,
            rrf_k,
            oversample,
            model,
            created_at
        FROM retrieval_runs
        ORDER BY run_id
    """).fetchall()

    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier2.retrieval_runs (
                    run_id,
                    concept,
                    from_year,
                    to_year,
                    seed_population,
                    neighbour_population,
                    scales,
                    top_n,
                    rrf_k,
                    oversample,
                    model,
                    created_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (run_id) DO NOTHING
                """,
                row,
            )


def migrate_concept_seeds(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            concept,
            from_year,
            to_year,
            event_id,
            role
        FROM concept_seeds
        ORDER BY concept, from_year, to_year, event_id
    """)


    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier2.concept_seeds (
                    concept,
                    from_year,
                    to_year,
                    event_id,
                    role
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                row,
            )


def migrate_event_field(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            concept,
            event_id,
            role
        FROM event_field
        ORDER BY concept, event_id
    """)

    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier2.event_field (
                    concept,
                    event_id,
                    role
                )
                VALUES (%s, %s, %s)
                ON CONFLICT (concept, event_id) DO UPDATE
                SET role = EXCLUDED.role
                """,
                row,
            )


def migrate_neighbour_edges(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            run_id,
            seed_event_id,
            neighbour_event_id,
            depth,
            via_event_id,
            rank,
            score,
            score_local,
            score_medium,
            score_broad
        FROM neighbour_edges
        ORDER BY run_id, seed_event_id, neighbour_event_id, depth
    """)

    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier2.neighbour_edges (
                    run_id,
                    seed_event_id,
                    neighbour_event_id,
                    depth,
                    via_event_id,
                    rank,
                    score,
                    score_local,
                    score_medium,
                    score_broad
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT DO NOTHING
                """,
                row,
            )


def migrate_concept_aggregate(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            concept,
            kind,
            rank,
            value,
            window_doc_id,
            window_id,
            count,
            score
        FROM concept_aggregate
        ORDER BY concept, kind, rank, id
    """)

    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier2.concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count,
                    score
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                row,
            )


def sync_run_sequence(pg: Connection) -> None:
    """
    The source run IDs are authoritative during migration.

    The sequence must advance beyond them before normal Tier 2 writes
    resume, otherwise a future generated run_id can collide.
    """
    with pg.cursor() as cur:
        cur.execute("""
            SELECT setval(
                pg_get_serial_sequence(
                    'tier2.retrieval_runs',
                    'run_id'
                ),
                COALESCE(
                    (SELECT MAX(run_id)
                     FROM tier2.retrieval_runs),
                    1
                ),
                true
            )
        """)


def validate_counts(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    tables = (
        "concepts",
        "concept_seeds",
        "retrieval_runs",
        "neighbour_edges",
        "event_field",
        "concept_aggregate",
    )

    for table in tables:
        source = sqlite_count(sqlite, table)
        target = postgres_count(pg, f"tier2.{table}")

        logger.info(
            "[tier2-migrate] %s: sqlite=%d postgres=%d",
            table,
            source,
            target,
        )

        if source != target:
            raise RuntimeError(
                f"count mismatch for {table}: "
                f"sqlite={source}, postgres={target}"
            )


def migrate(
    sqlite_path: Path,
    pg: Connection,
) -> None:
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)

    sqlite = sqlite3.connect(sqlite_path)

    try:
        logger.info(
            "[tier2-migrate] source=%s",
            sqlite_path,
        )

        create_tier2_schema(pg)

        validate_source_events(sqlite, pg)

        with pg.transaction():
            migrate_concepts(sqlite, pg)
            migrate_retrieval_runs(sqlite, pg)
            migrate_concept_seeds(sqlite, pg)
            migrate_event_field(sqlite, pg)
            migrate_neighbour_edges(sqlite, pg)
            migrate_concept_aggregate(sqlite, pg)
            sync_run_sequence(pg)

        validate_counts(sqlite, pg)

        logger.info("[tier2-migrate] migration successful")

    finally:
        sqlite.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
    )
    args = parser.parse_args()

    with get_connection() as pg:
        migrate(args.sqlite, pg)


if __name__ == "__main__":
    main()
