# src/tools/reset_clmet_tier1.py

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from tier1.vector_writer import lance_table_name


LANCE_BUCKET_SIZE = 50
LANCE_MODEL_NAME = "macberth"
SCALE = "medium"
DELETE_BATCH_SIZE = 5_000


def load_clmet_events(conn) -> list[tuple[int, str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                event_id,
                doc_id,
                pub_year
            FROM events
            WHERE corpus = 'clmet'
            ORDER BY event_id
            """
        )
        return [
            (int(event_id), doc_id, int(pub_year))
            for event_id, doc_id, pub_year in cur.fetchall()
        ]


def group_events_by_lance_table(
    events: list[tuple[int, str, int]],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)

    for event_id, _doc_id, pub_year in events:
        table_name = lance_table_name(
            scale=SCALE,
            model_name=LANCE_MODEL_NAME,
            year=pub_year,
            bucket_size=LANCE_BUCKET_SIZE,
        )
        grouped[table_name].append(event_id)

    return grouped


def delete_lance_events(
    db,
    events_by_table: dict[str, list[int]],
) -> None:
    for table_name, event_ids in sorted(events_by_table.items()):
        table = db.open_table(table_name)

        logger.info(
            "[reset] deleting %d CLMET vectors from %s",
            len(event_ids),
            table_name,
        )

        for start in range(0, len(event_ids), DELETE_BATCH_SIZE):
            batch = event_ids[start:start + DELETE_BATCH_SIZE]

            predicate = (
                "event_id IN ("
                + ",".join(str(event_id) for event_id in batch)
                + ")"
            )

            table.delete(predicate)


def verify_lance_deletion(
    db,
    events_by_table: dict[str, list[int]],
) -> int:
    remaining = 0

    for table_name, event_ids in sorted(events_by_table.items()):
        table = db.open_table(table_name)

        rows = table.search().select(["event_id"]).to_arrow()

        present = {
            int(event_id)
            for event_id in rows.column("event_id").to_pylist()
        }

        found = present.intersection(event_ids)

        if found:
            logger.error(
                "[reset] %d old CLMET vectors remain in %s",
                len(found),
                table_name,
            )
            remaining += len(found)

    return remaining


def delete_postgres_events(
    conn,
    event_ids: list[int],
) -> None:
    with conn.cursor() as cur:
        for start in range(0, len(event_ids), DELETE_BATCH_SIZE):
            batch = event_ids[start:start + DELETE_BATCH_SIZE]

            cur.execute(
                """
                DELETE FROM events
                WHERE corpus = 'clmet'
                  AND event_id = ANY(%s)
                """,
                (batch,),
            )

            logger.info(
                "[reset] deleted %d PostgreSQL events",
                cur.rowcount,
            )

        conn.commit()


def verify_postgres_deletion(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM events
            WHERE corpus = 'clmet'
            """
        )
        return int(cur.fetchone()[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the existing CLMET Tier 1 medium/MacBERTh "
            "events and vectors so CLMET can be regenerated cleanly."
        )
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path(LANCE_INDEXES_DIR),
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the CLMET Tier 1 data.",
    )

    args = parser.parse_args()

    with get_connection(
        application_name="reset-clmet-tier1",
    ) as conn:
        events = load_clmet_events(conn)

    print()
    print("=" * 72)
    print("CLMET TIER 1 RESET")
    print("=" * 72)
    print(f"PostgreSQL CLMET events: {len(events):,}")
    print(f"Lance scale:             {SCALE}")
    print(f"Lance model:             {LANCE_MODEL_NAME}")
    print(f"Lance bucket size:       {LANCE_BUCKET_SIZE}")
    print()

    if not events:
        print("No CLMET Tier 1 events exist in PostgreSQL.")
        print("Nothing to reset.")
        return

    events_by_table = group_events_by_lance_table(events)

    print("Affected Lance tables:")
    for table_name, event_ids in sorted(events_by_table.items()):
        print(f"  {table_name}: {len(event_ids):,}")
    print()

    if not args.execute:
        print("DRY RUN — nothing has been deleted.")
        print()
        print(
            "Run with --execute to remove these exact CLMET "
            "event IDs and vectors."
        )
        return

    db = lancedb.connect(str(args.lance_root))

    print("Deleting CLMET vectors from Lance...")
    delete_lance_events(db, events_by_table)

    print("Verifying Lance deletion...")
    remaining_lance = verify_lance_deletion(
        db,
        events_by_table,
    )

    if remaining_lance:
        raise RuntimeError(
            f"{remaining_lance:,} CLMET Lance vectors remain; "
            "PostgreSQL events were NOT deleted."
        )

    print("Lance deletion verified.")
    print()

    event_ids = [event_id for event_id, _, _ in events]

    with get_connection(
        application_name="reset-clmet-tier1",
    ) as conn:
        print("Deleting PostgreSQL CLMET events...")
        delete_postgres_events(conn, event_ids)

        remaining_pg = verify_postgres_deletion(conn)

    if remaining_pg:
        raise RuntimeError(
            f"{remaining_pg:,} CLMET PostgreSQL events remain."
        )

    print("PostgreSQL deletion verified.")
    print()
    print("=" * 72)
    print("CLMET TIER 1 RESET COMPLETE")
    print("=" * 72)
    print(f"Events removed: {len(events):,}")
    print("Lance vectors removed: verified")
    print("PostgreSQL events removed: verified")
    print()
    print(
        "The clean CLMET documents/tokens were not modified."
    )


if __name__ == "__main__":
    main()
