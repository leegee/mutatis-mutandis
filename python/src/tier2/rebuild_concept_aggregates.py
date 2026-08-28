"""
tier2/rebuild_concept_aggregates.py

Rebuild derived concept_aggregate rows from the existing Tier 2 SQLite data.

No LanceDB access or semantic retrieval is performed. The events and
neighbours tables remain unchanged.

Token and document counts represent distinct seed events that retrieved
the token/document, rather than raw retrieval multiplicity.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_logging import logger


def rebuild_concept_aggregates(
    db_path: Path,
    *,
    concept: str | None = None,
) -> None:
    conn = sqlite3.connect(db_path)

    try:
        conn.execute("PRAGMA foreign_keys = ON")

        if concept is None:
            concepts = [
                row[0]
                for row in conn.execute(
                    "SELECT concept FROM concepts ORDER BY concept"
                )
            ]
        else:
            concepts = [concept]

        for concept_name in concepts:
            logger.info(
                "[tier2 aggregates] rebuilding concept=%s",
                concept_name,
            )

            rows = conn.execute(
                """
                SELECT
                    n.event_id,
                    n.token,
                    n.doc_id,
                    n.window_id
                FROM neighbours n
                JOIN events e
                    ON e.event_id = n.event_id
                WHERE e.concept = ?
                """,
                (concept_name,),
            ).fetchall()

            token_seed_events: dict[str, set[int]] = defaultdict(set)
            doc_seed_events: dict[str, set[int]] = defaultdict(set)
            window_counts: dict[tuple[str, int], int] = defaultdict(int)

            for event_id, token, doc_id, window_id in rows:
                if token is not None:
                    token_seed_events[str(token)].add(
                        int(event_id)
                    )

                if doc_id is not None:
                    doc_id = str(doc_id)
                    doc_seed_events[doc_id].add(
                        int(event_id)
                    )

                    if window_id is not None:
                        window_counts[
                            (doc_id, int(window_id))
                        ] += 1

            conn.execute(
                """
                DELETE FROM concept_aggregate
                WHERE concept = ?
                """,
                (concept_name,),
            )

            aggregate_rows = []

            token_ranked = sorted(
                token_seed_events.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )

            for rank, (token, seed_events) in enumerate(
                token_ranked
            ):
                aggregate_rows.append(
                    (
                        concept_name,
                        "token",
                        rank,
                        token,
                        None,
                        None,
                        len(seed_events),
                    )
                )

            doc_ranked = sorted(
                doc_seed_events.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )

            for rank, (doc_id, seed_events) in enumerate(
                doc_ranked
            ):
                aggregate_rows.append(
                    (
                        concept_name,
                        "doc",
                        rank,
                        doc_id,
                        None,
                        None,
                        len(seed_events),
                    )
                )

            window_ranked = sorted(
                window_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for rank, ((doc_id, window_id), count) in enumerate(
                window_ranked
            ):
                aggregate_rows.append(
                    (
                        concept_name,
                        "window",
                        rank,
                        None,
                        doc_id,
                        window_id,
                        count,
                    )
                )

            conn.executemany(
                """
                INSERT INTO concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                aggregate_rows,
            )

            logger.info(
                "[tier2 aggregates] concept=%s: "
                "%d tokens, %d documents, %d windows",
                concept_name,
                len(token_ranked),
                len(doc_ranked),
                len(window_ranked),
            )

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Tier 2 concept aggregates from existing SQLite data."
    )

    parser.add_argument(
        "--sqlite",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
        help=(
            f"Tier 2 SQLite database "
            f"(default: {CORPUS_TIER2_DB_PATH})."
        ),
    )

    parser.add_argument(
        "--concept",
        help="Rebuild only this concept. Defaults to all concepts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rebuild_concept_aggregates(
        args.sqlite,
        concept=args.concept,
    )


if __name__ == "__main__":
    main()
