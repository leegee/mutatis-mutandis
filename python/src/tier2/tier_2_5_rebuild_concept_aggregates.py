"""
tier2/rebuild_concept_aggregates.py

Rebuild derived tier2.concept_aggregate rows from persisted Tier 2
retrieval data.

No LanceDB access or semantic retrieval is performed. Retrieval provenance
is authoritative in PostgreSQL:

    retrieval_runs -> neighbour_edges

Corpus provenance is authoritative in PostgreSQL:

    neighbour_edges.neighbour_event_id -> events.event_id

Aggregate ranking is based on accumulated RRF score. Counts represent
distinct seed-event contributions rather than raw retrieval multiplicity.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Iterable

from lib.corpus_db import get_connection
from lib.corpus_logging import logger


POSTGRES_BATCH_SIZE = 10_000


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


def _fetch_event_metadata(
    connection,
    event_ids: Iterable[int],
) -> dict[int, dict[str, Any]]:
    """
    Fetch authoritative Tier 1 provenance from PostgreSQL.

    Event IDs are batched so large Tier 2 retrievals do not create an
    unbounded query parameter list.
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


def _fetch_all_neighbour_edges(
    connection,
    concept: str,
) -> list[tuple[int, int, float]]:
    """
    Return persisted first-order retrieval edges for a concept.

    Each row is:

        seed_event_id, neighbour_event_id, score

    retrieval_runs identifies the concept; neighbour_edges supplies the
    persisted semantic retrieval evidence.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                ne.seed_event_id,
                ne.neighbour_event_id,
                ne.score
            FROM tier2.neighbour_edges AS ne
            JOIN tier2.retrieval_runs AS rr
              ON rr.run_id = ne.run_id
            WHERE rr.concept = %s
              AND ne.depth = 1
            ORDER BY ne.run_id, ne.seed_event_id, ne.rank
            """,
            (concept,),
        )

        rows = cursor.fetchall()

    return [
        (
            int(seed_event_id),
            int(neighbour_event_id),
            float(score),
        )
        for seed_event_id, neighbour_event_id, score in rows
        if score is not None
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
    Convert persisted retrieval edges and event metadata into aggregate data.

    A seed/neighbour pair contributes at most once. If duplicate persisted
    rows exist for the same pair, the highest score is retained.

    Counts therefore represent distinct seed-event contributions, while
    scores accumulate the retained retrieval scores.
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
        medium_window_id = event["medium_window_id"]

        token_seed_events[token].add(seed_event_id)
        token_scores[token] += score

        doc_seed_events[doc_id].add(seed_event_id)
        doc_scores[doc_id] += score

        if medium_window_id is not None:
            window_key = (doc_id, int(medium_window_id))

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


def _replace_concept_aggregates(
    connection,
    *,
    concept: str,
    token_ranked,
    doc_ranked,
    window_ranked,
) -> int:
    """
    Replace all derived aggregate rows for one concept atomically.
    """
    rows = []

    for rank, (
        token,
        seed_events,
        score,
    ) in enumerate(token_ranked):
        rows.append(
            (
                concept,
                "token",
                rank,
                token,
                None,
                None,
                len(seed_events),
                score,
            )
        )

    for rank, (
        doc_id,
        seed_events,
        score,
    ) in enumerate(doc_ranked):
        rows.append(
            (
                concept,
                "doc",
                rank,
                doc_id,
                None,
                None,
                len(seed_events),
                score,
            )
        )

    for rank, (
        (doc_id, window_id),
        seed_events,
        score,
    ) in enumerate(window_ranked):
        rows.append(
            (
                concept,
                "window",
                rank,
                None,
                doc_id,
                window_id,
                len(seed_events),
                score,
            )
        )

    with connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM tier2.concept_aggregate
            WHERE concept = %s
            """,
            (concept,),
        )

        if rows:
            with cursor.copy(
                """
                COPY tier2.concept_aggregate (
                    concept,
                    kind,
                    rank,
                    value,
                    window_doc_id,
                    window_id,
                    count,
                    score
                )
                FROM STDIN
                """
            ) as copy:
                for row in rows:
                    copy.write_row(row)

    return len(rows)


def _fetch_concepts(connection) -> list[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT concept
            FROM tier2.concepts
            ORDER BY concept
            """
        )

        return [str(row[0]) for row in cursor.fetchall()]


def rebuild_concept_aggregates(
    connection,
    *,
    concept: str | None = None,
) -> None:
    """
    Rebuild derived aggregates from the current PostgreSQL Tier 2 state.

    Each concept is rebuilt in its own transaction so a failure does not
    leave that concept with its previous aggregate rows deleted.
    """
    concepts = (
        [concept]
        if concept is not None
        else _fetch_concepts(connection)
    )

    for concept_name in concepts:
        logger.info(
            "[tier2 aggregates] rebuilding concept=%s",
            concept_name,
        )

        edges = _fetch_all_neighbour_edges(
            connection,
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
            postgres_connection=connection,
            edges=edges,
        )

        with connection.transaction():
            row_count = _replace_concept_aggregates(
                connection,
                concept=concept_name,
                token_ranked=token_ranked,
                doc_ranked=doc_ranked,
                window_ranked=window_ranked,
            )

        logger.info(
            "[tier2 aggregates] concept=%s: %d tokens, %d documents, %d windows, %d rows",
            concept_name,
            len(token_ranked),
            len(doc_ranked),
            len(window_ranked),
            row_count,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=( "Rebuild Tier 2 concept aggregates from persisted PostgreSQL retrieval data." )
    )

    parser.add_argument(
        "--concept",
        help="Rebuild only this concept. Defaults to all concepts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    connection = get_connection()

    try:
        rebuild_concept_aggregates(
            connection,
            concept=args.concept,
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
