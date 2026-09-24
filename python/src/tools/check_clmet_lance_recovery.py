# src/tools/check_clmet_lance_recovery.py

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from tier1.vector_writer import lance_table_name, year_bucket

LANCE_BUCKET_SIZE = 50
LANCE_MODEL_NAME = "macberth"
SCALE = "medium"


def load_clmet_events(conn) -> list[tuple[int, str, int | None]]:
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
        return cur.fetchall()


def load_lance_event_ids(
    db,
    *,
    pub_year: int,
) -> set[int]:
    table_name = lance_table_name(
        scale=SCALE,
        model_name=LANCE_MODEL_NAME,
        year=pub_year,
        bucket_size=LANCE_BUCKET_SIZE,
    )

    try:
        table = db.open_table(table_name)
    except Exception:
        return set()

    rows = table.search().select(["event_id"]).to_arrow()

    return {
        int(event_id)
        for event_id in rows.column("event_id").to_pylist()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CLMET PostgreSQL Tier 1 events with medium-scale "
            "MacBERTh vectors in Lance without modifying either store."
        )
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=Path(LANCE_INDEXES_DIR),
    )

    args = parser.parse_args()

    with get_connection(
        application_name="check-clmet-lance-recovery",
    ) as conn:
        events = load_clmet_events(conn)

    if not events:
        raise RuntimeError("No CLMET events found in PostgreSQL.")

    logger.info(
        "[recovery] PostgreSQL CLMET events: %d",
        len(events),
    )

    by_year: dict[int, list[tuple[int, str]]] = defaultdict(list)

    for event_id, doc_id, pub_year in events:
        if pub_year is None:
            raise RuntimeError(
                f"CLMET event {event_id} has no publication year."
            )

        by_year[pub_year].append(
            (int(event_id), doc_id)
        )

    db = lancedb.connect(str(args.lance_root))

    present: set[int] = set()

    for pub_year, year_events in sorted(by_year.items()):
        lance_ids = load_lance_event_ids(
            db,
            pub_year=pub_year,
        )

        expected = {
            event_id
            for event_id, _ in year_events
        }

        found = expected & lance_ids
        missing = expected - lance_ids

        present.update(found)

        logger.info(
            "[recovery] %d: PostgreSQL=%d Lance=%d found=%d missing=%d",
            pub_year,
            len(expected),
            len(lance_ids),
            len(found),
            len(missing),
        )

    expected_ids = {
        event_id
        for event_id, _, _ in events
    }

    missing_ids = expected_ids - present
    unexpected_ids = present - expected_ids

    print()
    print("=" * 72)
    print("CLMET TIER 1 / LANCE RECOVERY CHECK")
    print("=" * 72)
    print(f"PostgreSQL CLMET events:       {len(expected_ids):,}")
    print(f"Lance vectors found:           {len(present):,}")
    print(f"Missing Lance vectors:         {len(missing_ids):,}")
    print(f"Unexpected matching vectors:   {len(unexpected_ids):,}")
    print()

    if not missing_ids:
        print("RESULT: PostgreSQL and Lance are complete for CLMET.")
        print("No repair is required.")
        return

    event_to_doc = {
        int(event_id): doc_id
        for event_id, doc_id, _ in events
    }

    missing_by_doc: dict[str, list[int]] = defaultdict(list)

    for event_id in sorted(missing_ids):
        missing_by_doc[event_to_doc[event_id]].append(event_id)

    print("Missing vectors by document:")
    print()

    for doc_id, doc_event_ids in sorted(
        missing_by_doc.items(),
        key=lambda item: min(item[1]),
    ):
        print(
            f"{doc_id:15s} "
            f"{len(doc_event_ids):8,} missing "
            f"event IDs "
            f"{min(doc_event_ids)}–{max(doc_event_ids)}"
        )

    print()
    print(
        "Do not rerun normal corpus processing until these missing "
        "vectors have been examined."
    )


if __name__ == "__main__":
    main()