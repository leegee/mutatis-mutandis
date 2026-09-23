# src/tools/clmet-lance-dry-run.py

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection


TABLES = (
    "medium__macberth__1700_1749",
    "medium__macberth__1750_1799",
    "medium__macberth__1800_1849",
    "medium__macberth__1850_1899",
    "medium__macberth__1900_1949",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run validation of CLMET event IDs against "
            "the corresponding Lance buckets. No data is modified."
        )
    )
    parser.add_argument(
        "--lance-root",
        type=Path,
        default=LANCE_INDEXES_DIR,
    )
    return parser.parse_args()


def load_clmet_events(con) -> dict[str, set[int]]:
    with con.cursor() as cur:
        cur.execute(
            """
            SELECT event_id, pub_year
            FROM events
            WHERE corpus = 'clmet'
            ORDER BY event_id
            """
        )

        events_by_table: dict[str, set[int]] = defaultdict(set)

        for event_id, pub_year in cur.fetchall():
            if pub_year is None:
                raise RuntimeError(
                    f"CLMET event {event_id} has NULL pub_year"
                )

            bucket_start = (int(pub_year) // 50) * 50
            table_name = (
                f"medium__macberth__"
                f"{bucket_start:04d}_{bucket_start + 49:04d}"
            )

            events_by_table[table_name].add(int(event_id))

    return dict(events_by_table)


def load_lance_event_ids(table, *, batch_size: int = 50_000) -> set[int]:
    total_rows = table.count_rows()
    event_ids: set[int] = set()

    offset = 0

    while offset < total_rows:
        rows = (
            table.search()
            .select(["event_id"])
            .limit(batch_size)
            .offset(offset)
            .to_list()
        )

        if not rows:
            break

        event_ids.update(
            int(row["event_id"])
            for row in rows
        )

        offset += len(rows)

    return event_ids


def inspect_table(
    db,
    table_name: str,
    expected_clmet_ids: set[int],
) -> None:
    print("=" * 72)
    print(table_name)

    if table_name not in db.list_tables().tables:
        print("  ERROR: Lance table does not exist")
        return

    table = db.open_table(table_name)

    lance_ids = load_lance_event_ids(table)

    expected_count = len(expected_clmet_ids)
    lance_count = len(lance_ids)

    clmet_in_lance = lance_ids & expected_clmet_ids
    missing = expected_clmet_ids - lance_ids

    # The Lance table contains all corpora for mixed buckets. Any Lance
    # event ID not belonging to CLMET is therefore something we must preserve.
    non_clmet_in_lance = lance_ids - expected_clmet_ids

    print(f"  Lance rows:              {lance_count:,}")
    print(f"  PostgreSQL CLMET rows:   {expected_count:,}")
    print(f"  CLMET rows found:        {len(clmet_in_lance):,}")
    print(f"  CLMET rows missing:      {len(missing):,}")
    print(
        f"  Non-CLMET rows preserved: "
        f"{len(non_clmet_in_lance):,}"
    )

    if missing:
        print()
        print("  ERROR: CLMET events present in PostgreSQL but absent from Lance")

        sample = sorted(missing)[:20]
        print(f"  First missing IDs: {sample}")

    if len(clmet_in_lance) != expected_count:
        print()
        print("  ERROR: CLMET event count does not match")

    if expected_count + len(non_clmet_in_lance) != lance_count:
        print()
        print("  ERROR: Lance rows cannot be partitioned cleanly")

    if (
        len(clmet_in_lance) == expected_count
        and expected_count + len(non_clmet_in_lance) == lance_count
    ):
        print("  STATUS: OK")
        print(
            f"  WOULD DELETE: {len(clmet_in_lance):,} Lance rows"
        )

    print()


def main() -> None:
    args = parse_args()

    db = lancedb.connect(str(args.lance_root))
    con = get_connection()

    try:
        print("CLMET Lance deletion dry run")
        print("============================")
        print(f"Lance root: {args.lance_root}")
        print()
        print("NO DATA WILL BE MODIFIED.")
        print()

        events_by_table = load_clmet_events(con)

        total_clmet = sum(
            len(event_ids)
            for event_ids in events_by_table.values()
        )

        print(
            f"PostgreSQL CLMET events: {total_clmet:,}"
        )
        print()

        expected_tables = set(TABLES)
        actual_tables = set(events_by_table)

        unexpected_tables = actual_tables - expected_tables

        if unexpected_tables:
            print(
                "ERROR: CLMET events occur in unexpected Lance buckets:"
            )
            for table_name in sorted(unexpected_tables):
                print(
                    f"  {table_name}: "
                    f"{len(events_by_table[table_name]):,}"
                )
            print()

        for table_name in TABLES:
            inspect_table(
                db,
                table_name,
                events_by_table.get(table_name, set()),
            )

        print("=" * 72)
        print("Summary")
        print("=" * 72)

        for table_name in TABLES:
            count = len(
                events_by_table.get(table_name, set())
            )
            print(f"{table_name}: {count:,} CLMET rows")

        print()
        print(f"TOTAL WOULD DELETE: {total_clmet:,}")
        print()
        print("Dry run complete. No data was modified.")

    finally:
        con.close()


if __name__ == "__main__":
    main()
