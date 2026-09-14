# migrate_tier3_sqlite_to_postgres.py

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from psycopg import Connection

from lib.corpus_config import CORPUS_TIER3_DB_PATH
from lib.corpus_db import create_tier3_schema, get_connection
from lib.corpus_logging import logger


BATCH_SIZE = 10_000


def sqlite_count(
    conn: sqlite3.Connection,
    table: str,
) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table}"
    ).fetchone()

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


def validate_event_ids(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    logger.info("[tier3-migrate] validating event references")

    event_ids = {
        int(row[0])
        for row in sqlite.execute("""
            SELECT DISTINCT event_id
            FROM event_geometry
        """)
    }

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

            found = {
                int(row[0])
                for row in cur.fetchall()
            }

            missing.update(set(batch) - found)

    if missing:
        sample = sorted(missing)[:20]
        raise RuntimeError(
            f"event_geometry references {len(missing)} "
            f"events absent from PostgreSQL events; "
            f"sample={sample}"
        )

    logger.info(
        "[tier3-migrate] validated %d event IDs",
        len(event_ids),
    )


def validate_concepts(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    concepts = {
        str(row[0])
        for row in sqlite.execute("""
            SELECT DISTINCT concept
            FROM event_geometry
        """)
    }

    with pg.cursor() as cur:
        cur.execute(
            """
            SELECT concept
            FROM tier2.concepts
            WHERE concept = ANY(%s)
            """,
            (list(concepts),),
        )

        found = {
            str(row[0])
            for row in cur.fetchall()
        }

    missing = concepts - found

    if missing:
        raise RuntimeError(
            "Tier 3 references concepts absent from tier2.concepts: "
            f"{sorted(missing)}"
        )


def migrate_event_geometry(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            concept,
            event_id,
            nx,
            ny,
            gnx,
            gny,
            cluster_id,
            cluster_label
        FROM event_geometry
        ORDER BY concept, event_id
    """)

    count = 0

    with pg.cursor() as cur:
        for row in rows:
            cur.execute(
                """
                INSERT INTO tier3.event_geometry (
                    concept,
                    event_id,
                    nx,
                    ny,
                    gnx,
                    gny,
                    cluster_id,
                    cluster_label
                )
                VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                ON CONFLICT (concept, event_id)
                DO UPDATE SET
                    nx = EXCLUDED.nx,
                    ny = EXCLUDED.ny,
                    gnx = EXCLUDED.gnx,
                    gny = EXCLUDED.gny,
                    cluster_id = EXCLUDED.cluster_id,
                    cluster_label = EXCLUDED.cluster_label
                """,
                row,
            )

            count += 1

    logger.info(
        "[tier3-migrate] migrated event_geometry rows=%d",
        count,
    )


def migrate_cluster_info(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    rows = sqlite.execute("""
        SELECT
            concept,
            cluster_id,
            cluster_label,
            centroid_nx,
            centroid_ny,
            centroid_gnx,
            centroid_gny,
            centroid_vector,
            point_count,
            description
        FROM concept_cluster_info
        ORDER BY concept, cluster_id
    """)

    count = 0

    with pg.cursor() as cur:
        for row in rows:
            (
                concept,
                cluster_id,
                cluster_label,
                centroid_nx,
                centroid_ny,
                centroid_gnx,
                centroid_gny,
                centroid_vector,
                point_count,
                description,
            ) = row

            if centroid_vector is None:
                raise RuntimeError(
                    f"NULL centroid_vector for "
                    f"{concept}/{cluster_id}"
                )

            cur.execute(
                """
                INSERT INTO tier3.concept_cluster_info (
                    concept,
                    cluster_id,
                    cluster_label,
                    centroid_nx,
                    centroid_ny,
                    centroid_gnx,
                    centroid_gny,
                    centroid_vector,
                    point_count,
                    description
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s
                )
                ON CONFLICT (concept, cluster_id)
                DO UPDATE SET
                    cluster_label = EXCLUDED.cluster_label,
                    centroid_nx = EXCLUDED.centroid_nx,
                    centroid_ny = EXCLUDED.centroid_ny,
                    centroid_gnx = EXCLUDED.centroid_gnx,
                    centroid_gny = EXCLUDED.centroid_gny,
                    centroid_vector = EXCLUDED.centroid_vector,
                    point_count = EXCLUDED.point_count,
                    description = EXCLUDED.description
                """,
                row,
            )

            count += 1

    logger.info(
        "[tier3-migrate] migrated cluster_info rows=%d",
        count,
    )


def validate_counts(
    sqlite: sqlite3.Connection,
    pg: Connection,
) -> None:
    for table in (
        "event_geometry",
        "concept_cluster_info",
    ):
        source = sqlite_count(sqlite, table)
        target = postgres_count(
            pg,
            f"tier3.{table}",
        )

        logger.info(
            "[tier3-migrate] %s: sqlite=%d postgres=%d",
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
        raise FileNotFoundError(
            f"Tier 3 database does not exist: {sqlite_path}"
        )

    sqlite = sqlite3.connect(sqlite_path)

    try:
        logger.info(
            "[tier3-migrate] source=%s",
            sqlite_path,
        )

        create_tier3_schema(pg)

        validate_event_ids(sqlite, pg)
        validate_concepts(sqlite, pg)

        with pg.transaction():
            migrate_event_geometry(sqlite, pg)
            migrate_cluster_info(sqlite, pg)

        validate_counts(sqlite, pg)

        logger.info(
            "[tier3-migrate] migration successful"
        )

    finally:
        sqlite.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path(CORPUS_TIER3_DB_PATH),
    )

    args = parser.parse_args()

    with get_connection() as pg:
        migrate(args.sqlite, pg)


if __name__ == "__main__":
    main()