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

MIXED_TABLES = {
    "medium__macberth__1700_1749",
    "medium__macberth__1750_1799",
}

BATCH_SIZE = 5_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete existing CLMET embeddings from the corresponding "
            "Lance buckets after complete PostgreSQL/Lance reconciliation."
        )
    )
    parser.add_argument(
        "--lance-root",
        type=Path,
        default=LANCE_INDEXES_DIR,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows. Without this flag nothing is modified.",
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


def load_lance_event_ids(
    table,
    *,
    batch_size: int = 50_000,
) -> set[int]:
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


def validate_table(
    db,
    table_name: str,
    expected_clmet_ids: set[int],
) -> tuple[object, set[int], set[int]]:
    print("=" * 72)
    print(table_name)

    if table_name not in db.list_tables().tables:
        raise RuntimeError(
            f"Lance table does not exist: {table_name}"
        )

    table = db.open_table(table_name)

    lance_ids = load_lance_event_ids(table)

    expected_count = len(expected_clmet_ids)
    lance_count = len(lance_ids)

    clmet_in_lance = lance_ids & expected_clmet_ids
    missing = expected_clmet_ids - lance_ids

    # In mixed buckets these are EEBO/ECCO rows that must survive deletion.
    preserved_ids = lance_ids - expected_clmet_ids

    print(f"  Lance rows:               {lance_count:,}")
    print(f"  PostgreSQL CLMET rows:    {expected_count:,}")
    print(f"  CLMET rows found:         {len(clmet_in_lance):,}")
    print(f"  CLMET rows missing:       {len(missing):,}")
    print(f"  Non-CLMET rows preserved: {len(preserved_ids):,}")

    if missing:
        sample = sorted(missing)[:20]
        raise RuntimeError(
            f"{table_name}: PostgreSQL CLMET events are missing "
            f"from Lance. First missing IDs: {sample}"
        )

    if clmet_in_lance != expected_clmet_ids:
        raise RuntimeError(
            f"{table_name}: CLMET event reconciliation failed"
        )

    if expected_count + len(preserved_ids) != lance_count:
        raise RuntimeError(
            f"{table_name}: Lance rows cannot be partitioned cleanly"
        )

    print("  STATUS: OK")
    print(f"  WOULD DELETE:             {len(clmet_in_lance):,}")
    print()

    return table, clmet_in_lance, preserved_ids


def delete_event_ids(
    table,
    event_ids: set[int],
    *,
    batch_size: int = BATCH_SIZE,
) -> None:
    ordered_ids = sorted(event_ids)
    total = len(ordered_ids)

    deleted_so_far = 0

    for start in range(0, total, batch_size):
        batch = ordered_ids[start:start + batch_size]

        predicate = (
            "event_id IN ("
            + ", ".join(str(event_id) for event_id in batch)
            + ")"
        )

        table.delete(predicate)

        deleted_so_far += len(batch)

        print(
            f"    deletion batches: "
            f"{deleted_so_far:,}/{total:,}",
            flush=True,
        )


def delete_all_rows(table, expected_count: int) -> None:
    # event_id is non-null for every corpus event, so this predicate clears
    # a CLMET-only bucket without constructing a multi-million-ID predicate.
    table.delete("event_id IS NOT NULL")

    remaining = table.count_rows()

    if remaining != 0:
        raise RuntimeError(
            f"Expected empty Lance table after deletion, "
            f"but {remaining:,} rows remain"
        )

    print(
        f"    deleted {expected_count:,} rows; "
        f"0 remain",
        flush=True,
    )


def confirm_deletion(total_rows: int) -> None:
    print()
    print("=" * 72)
    print("DESTRUCTIVE OPERATION")
    print("=" * 72)
    print(
        f"This will delete {total_rows:,} existing CLMET Lance rows."
    )
    print()
    print("PostgreSQL will NOT be modified.")
    print("EEBO/ECCO rows in mixed buckets will be preserved.")
    print()
    print(
        "Type DELETE CLMET to continue: ",
        end="",
        flush=True,
    )

    confirmation = input().strip()

    if confirmation != "DELETE CLMET":
        raise RuntimeError(
            "Deletion cancelled: confirmation did not match"
        )

    print()


def main() -> None:
    args = parse_args()

    print("CLMET Lance deletion")
    print("====================")
    print(f"Lance root: {args.lance_root}")
    print()

    if not args.execute:
        print("DRY RUN: no data will be modified.")
    else:
        print("EXECUTE MODE: Lance data will be modified.")

    print()

    db = lancedb.connect(str(args.lance_root))
    con = get_connection(
        application_name="clmet-lance-delete",
    )

    try:
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
            raise RuntimeError(
                "CLMET events occur in unexpected Lance buckets: "
                + ", ".join(sorted(unexpected_tables))
            )

        validated: dict[
            str,
            tuple[object, set[int], set[int]],
        ] = {}

        # Complete validation happens before the first mutation.
        for table_name in TABLES:
            validated[table_name] = validate_table(
                db,
                table_name,
                events_by_table.get(
                    table_name,
                    set(),
                ),
            )

        print("=" * 72)
        print("Pre-deletion validation complete")
        print("=" * 72)
        print()

        for table_name in TABLES:
            _, _, preserved_ids = validated[table_name]

            print(
                f"{table_name}: "
                f"{len(events_by_table.get(table_name, set())):,} "
                f"CLMET rows"
            )
            print(
                f"  {len(preserved_ids):,} non-CLMET rows will remain"
            )

        print()
        print(f"TOTAL CLMET ROWS: {total_clmet:,}")
        print()

        if not args.execute:
            print("Dry run complete. No data was modified.")
            return

        confirm_deletion(total_clmet)

        print("Beginning deletion...")
        print()

        for table_name in TABLES:
            table, clmet_ids, preserved_ids = validated[table_name]

            print("=" * 72)
            print(table_name)
            print("=" * 72)

            expected_delete = len(clmet_ids)

            if table_name in MIXED_TABLES:
                print(
                    f"Deleting {expected_delete:,} CLMET rows "
                    f"in batches of {BATCH_SIZE:,}..."
                )

                delete_event_ids(
                    table,
                    clmet_ids,
                )

            else:
                print(
                    f"Clearing {expected_delete:,} CLMET-only rows..."
                )

                delete_all_rows(
                    table,
                    expected_delete,
                )

            # Reload the table after mutation and require the exact
            # precomputed survivor set. This catches accidental deletion
            # of EEBO/ECCO rows as well as incomplete CLMET deletion.
            remaining_ids = load_lance_event_ids(table)

            if remaining_ids != preserved_ids:
                missing = preserved_ids - remaining_ids
                unexpected = remaining_ids - preserved_ids

                raise RuntimeError(
                    f"{table_name}: post-delete verification failed: "
                    f"{len(missing):,} expected survivors missing; "
                    f"{len(unexpected):,} unexpected rows present"
                )

            print(
                f"  DELETED:   {expected_delete:,}"
            )
            print(
                f"  REMAINING: {len(remaining_ids):,}"
            )
            print("  STATUS:    OK")
            print()

        print("=" * 72)
        print("DELETION COMPLETE")
        print("=" * 72)
        print()
        print("Expected surviving Lance rows:")
        print("  1700-1749:    5,067")
        print("  1750-1799:   16,593")
        print("  1800-1849:        0")
        print("  1850-1899:        0")
        print("  1900-1949:        0")
        print()
        print("PostgreSQL was not modified.")
        print("Lance vector indexes were not rebuilt.")

    finally:
        con.close()


if __name__ == "__main__":
    main()