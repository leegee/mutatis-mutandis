# tier1/remove_orphaned_lance_events.py

from __future__ import annotations

import argparse
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from tier1.tier1_corpus2events import ACTIVE_SCALES, LANCE_MODEL_NAME


DEFAULT_BATCH_SIZE = 10_000
DELETE_BATCH_SIZE = 1_000


def active_table_names(
    db,
) -> list[str]:
    prefixes = tuple(
        f"{scale}__{LANCE_MODEL_NAME}__"
        for scale in ACTIVE_SCALES
    )

    return sorted(
        name
        for name in db.list_tables().tables
        if name.startswith(prefixes)
    )


def lance_event_ids(
    table,
    batch_size: int,
):
    # The audit only needs event_id. Keeping the scan columnar avoids
    # materialising vectors, which would turn a metadata reconciliation
    # into a large memory allocation.
    arrow = table.to_arrow()

    values = arrow.column("event_id").to_pylist()

    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


def postgres_event_ids(
    conn,
    event_ids: list[int],
) -> set[int]:
    if not event_ids:
        return set()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT event_id
            FROM events
            WHERE event_id = ANY(%s)
            """,
            (event_ids,),
        )

        return {
            row[0]
            for row in cur.fetchall()
        }


def find_orphans(
    conn,
    table,
    *,
    batch_size: int,
) -> list[int]:
    orphans: list[int] = []

    for batch in lance_event_ids(
        table,
        batch_size,
    ):
        existing = postgres_event_ids(
            conn,
            batch,
        )

        orphans.extend(
            event_id
            for event_id in batch
            if event_id not in existing
        )

    return orphans


def delete_orphans(
    table,
    event_ids: list[int],
    *,
    batch_size: int,
) -> int:
    deleted = 0

    for offset in range(0, len(event_ids), batch_size):
        batch = event_ids[
            offset : offset + batch_size
        ]

        # event_id is generated from a 63-bit hash in Tier 1, so the
        # predicate can safely use integer literals rather than relying
        # on signed/unsigned database coercion.
        predicate = (
            "event_id IN ("
            + ",".join(str(event_id) for event_id in batch)
            + ")"
        )

        table.delete(predicate)
        deleted += len(batch)

        logger.info(
            "[tier1] deleted %d/%d orphan rows",
            deleted,
            len(event_ids),
        )

    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove Lance event vectors whose event_id is absent "
            "from authoritative PostgreSQL events."
        )
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path(LANCE_INDEXES_DIR),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete orphaned Lance rows.",
    )

    parser.add_argument(
        "--table",
        action="append",
        help="Restrict the audit to one or more Lance table names.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    return args


def main() -> None:
    args = parse_args()

    db = lancedb.connect(str(args.lance_root))
    conn = get_connection()

    try:
        names = active_table_names(db)

        if args.table:
            requested = set(args.table)
            names = [
                name
                for name in names
                if name in requested
            ]

        logger.info(
            "[tier1] orphan audit: %d Lance table(s)",
            len(names),
        )

        total_rows = 0
        total_orphans = 0

        for table_name in names:
            table = db.open_table(table_name)
            row_count = table.count_rows()

            logger.info(
                "[tier1] scanning %s (%d rows)",
                table_name,
                row_count,
            )

            orphans = find_orphans(
                conn,
                table,
                batch_size=args.batch_size,
            )

            total_rows += row_count
            total_orphans += len(orphans)

            logger.info(
                "[tier1] %s: rows=%d orphans=%d",
                table_name,
                row_count,
                len(orphans),
            )

            if args.delete and orphans:
                deleted = delete_orphans(
                    table,
                    orphans,
                    batch_size=DELETE_BATCH_SIZE,
                )

                logger.info(
                    "[tier1] %s: deleted=%d",
                    table_name,
                    deleted,
                )

        logger.info(
            "[tier1] orphan audit complete: "
            "rows=%d orphans=%d mode=%s",
            total_rows,
            total_orphans,
            "DELETE" if args.delete else "DRY-RUN",
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()

