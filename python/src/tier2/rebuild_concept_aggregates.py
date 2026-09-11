"""
tier2/rebuild_concept_aggregates.py

Rebuild derived concept_aggregate rows from persisted Tier 2 retrieval data.

No LanceDB access or semantic retrieval is performed. The retrieval_runs and
neighbour_edges tables remain unchanged.

Retrieval provenance is authoritative in SQLite:
    retrieval_runs -> neighbour_edges

Corpus provenance is authoritative in PostgreSQL:
    neighbour_edges.neighbour_event_id -> events.event_id

Aggregate ranking is based on accumulated RRF score. Counts represent
distinct seed-event contributions, rather than raw retrieval multiplicity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_db import analysis_db_connection, get_connection
from lib.corpus_logging import logger


POSTGRES_BATCH_SIZE = 10_000


def _fetch_event_metadata(
    connection,
    event_ids: Iterable[int],
) -> dict[int, dict[str, Any]]:
    """
    Fetch authoritative Tier 1 provenance from PostgreSQL in one bounded query.

    The caller is responsible for batching event_ids so the query remains
    bounded for large Tier 2 retrievals.
    """
    ids = [int(event_id) for event_id in event_ids]

    if not ids:
        return {}

    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                event_id,
                doc_id,
                token,
                token_idx,
                pub_year,
                local_window_id,
                local_window_token_pos,
                medium_window_id,
                medium_window_token_pos,
                broad_window_id,
                broad_window_token_pos
            FROM events
            WHERE event_id = ANY(%s)
            """,
            (ids,),
        )

        rows = cursor.fetchall()

    metadata: dict[int, dict[str, Any]] = {}

    for row in rows:
        (
            event_id,
            doc_id,
            token,
            token_idx,
            pub_year,
            local_window_id,
            local_window_token_pos,
            medium_window_id,
            medium_window_token_pos,
            broad_window_id,
            broad_window_token_pos,
        ) = row

        metadata[int(event_id)] = {
            "event_id": int(event_id),
            "doc_id": str(doc_id),
            "token": str(token),
            "token_idx": int(token_idx),
            "pub_year": int(pub_year),
            "local_window_id": local_window_id,
            "local_window_token_pos": local_window_token_pos,
            "medium_window_id": medium_window_id,
            "medium_window_token_pos": medium_window_token_pos,
            "broad_window_id": broad_window_id,
            "broad_window_token_pos": broad_window_token_pos,
        }

    missing = set(ids) - set(metadata)

    if missing:
        raise RuntimeError(
            "Tier 2 requested event IDs absent from PostgreSQL: "
            f"{sorted(missing)[:10]}"
        )

    return metadata


def _batched(
    values: Iterable[int],
    batch_size: int,
):
    batch: list[int] = []

    for value in values:
        batch.append(int(value))

        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _fetch_all_neighbour_edges(
    conn,
    concept: str,
) -> list[tuple[int, int, float]]:
    """
    Return persisted first-order retrieval edges for a concept.

    Each row is:
        seed_event_id, neighbour_event_id, score

    Retrieval runs define the concept and interval. The edge table supplies
    the actual semantic retrieval evidence.
    """
    rows = conn.execute(
        """
        SELECT
            ne.seed_event_id,
            ne.neighbour_event_id,
            ne.score
        FROM neighbour_edges ne
        JOIN retrieval_runs rr
            ON rr.run_id = ne.run_id
        WHERE rr.concept = ?
          AND ne.depth = 1
        ORDER BY ne.run_id, ne.seed_event_id, ne.rank
        """,
        (concept,),
    ).fetchall()

    return [
        (
            int(seed_event_id),
            int(neighbour_event_id),
            float(score),
        )
        for seed_event_id, neighbour_event_id, score in rows
    ]


def _build_aggregates(
    *,
    postgres_connection,
    edges: list[tuple[int, int, float]],
) -> tuple[
    list[tuple[str, set[int], float]],
    list[tuple[str, set[int], float]],
    list[tuple[tuple[str, int], set[int], float]],
]:
    """
    Convert retrieval edges plus PostgreSQL event metadata into aggregate data.

    The sets preserve distinct seed-event provenance. Scores are accumulated
    across distinct seed/neighbour contributions.

    If the same seed retrieves the same neighbour more than once in persisted
    data, only one contribution is retained for that seed/neighbour pair.
    """
    unique_edges: dict[tuple[int, int], float] = {}

    for seed_event_id, neighbour_event_id, score in edges:
        key = (seed_event_id, neighbour_event_id)

        existing = unique_edges.get(key)

        if existing is None or score > existing:
            unique_edges[key] = score

    neighbour_ids = {
        neighbour_event_id
        for _, neighbour_event_id in unique_edges
    }

    metadata: dict[int, dict[str, Any]] = {}

    for batch in _batched(
        neighbour_ids,
        POSTGRES_BATCH_SIZE,
    ):
        metadata.update(
            _fetch_event_metadata(
                postgres_connection,
                batch,
            )
        )

    token_seed_events: dict[str, set[int]] = defaultdict(set)
    token_scores: dict[str, float] = defaultdict(float)

    doc_seed_events: dict[str, set[int]] = defaultdict(set)
    doc_scores: dict[str, float] = defaultdict(float)

    window_seed_events: dict[tuple[str, int], set[int]] = defaultdict(set)
    window_scores: dict[tuple[str, int], float] = defaultdict(float)

    for (seed_event_id, neighbour_event_id), score in unique_edges.items():
        event = metadata[neighbour_event_id]

        token = str(event["token"])
        doc_id = str(event["doc_id"])
        local_window_id = event["local_window_id"]

        token_seed_events[token].add(seed_event_id)
        token_scores[token] += score

        doc_seed_events[doc_id].add(seed_event_id)
        doc_scores[doc_id] += score

        if local_window_id is not None:
            window_key = (doc_id, int(local_window_id))

            window_seed_events[window_key].add(seed_event_id)
            window_scores[window_key] += score

    token_ranked = sorted(
        token_seed_events,
        key=lambda token: (
            -token_scores[token],
            -len(token_seed_events[token]),
            token,
        ),
    )

    doc_ranked = sorted(
        doc_seed_events,
        key=lambda doc_id: (
            -doc_scores[doc_id],
            -len(doc_seed_events[doc_id]),
            doc_id,
        ),
    )

    window_ranked = sorted(
        window_seed_events,
        key=lambda window: (
            -window_scores[window],
            -len(window_seed_events[window]),
            window[0],
            window[1],
        ),
    )

    return (
        [
            (
                token,
                token_seed_events[token],
                token_scores[token],
            )
            for token in token_ranked
        ],
        [
            (
                doc_id,
                doc_seed_events[doc_id],
                doc_scores[doc_id],
            )
            for doc_id in doc_ranked
        ],
        [
            (
                window,
                window_seed_events[window],
                window_scores[window],
            )
            for window in window_ranked
        ],
    )


def rebuild_concept_aggregates(
    db_path: Path,
    *,
    concept: str | None = None,
) -> None:
    sqlite_conn = analysis_db_connection(db_path)
    postgres_conn = get_connection()

    try:
        sqlite_conn.execute("PRAGMA foreign_keys = ON")

        if concept is None:
            concepts = [
                str(row[0])
                for row in sqlite_conn.execute(
                    """
                    SELECT concept
                    FROM concepts
                    ORDER BY concept
                    """
                )
            ]
        else:
            concepts = [concept]

        for concept_name in concepts:
            logger.info(
                "[tier2 aggregates] rebuilding concept=%s",
                concept_name,
            )

            edges = _fetch_all_neighbour_edges(
                sqlite_conn,
                concept_name,
            )

            logger.info(
                "[tier2 aggregates] concept=%s: %d persisted edges",
                concept_name,
                len(edges),
            )

            (
                token_ranked,
                doc_ranked,
                window_ranked,
            ) = _build_aggregates(
                postgres_connection=postgres_conn,
                edges=edges,
            )

            sqlite_conn.execute(
                """
                DELETE FROM concept_aggregate
                WHERE concept = ?
                """,
                (concept_name,),
            )

            aggregate_rows = []

            for rank, (
                token,
                seed_events,
                score,
            ) in enumerate(token_ranked):
                aggregate_rows.append(
                    (
                        concept_name,
                        "token",
                        rank,
                        token,
                        None,
                        None,
                        len(seed_events),
                        score
                    )
                )

            for rank, (
                doc_id,
                seed_events,
                score,
            ) in enumerate(doc_ranked):
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

            for rank, (
                (doc_id, window_id),
                seed_events,
                score,
            ) in enumerate(window_ranked):
                aggregate_rows.append(
                    (
                        concept_name,
                        "window",
                        rank,
                        None,
                        doc_id,
                        window_id,
                        len(seed_events),
                    )
                )

            sqlite_conn.executemany(
                """
                INSERT INTO concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count,
                    seed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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

        sqlite_conn.commit()

    except Exception:
        sqlite_conn.rollback()
        raise

    finally:
        postgres_conn.close()
        sqlite_conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Tier 2 concept aggregates from persisted "
            "retrieval data."
        )
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