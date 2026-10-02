#!/usr/bin/env python3
"""
Find corpus-wide semantic neighbours of an actual Campbell Apocalypse
MacBERTh observation.

The query is an existing Lance vector reconstructed from PostgreSQL/Lance
provenance; no new phrase embedding is generated.

By default the second occurrence of "whyte" in the Campbell passage is used.

Examples:

    python src/campbell_neighbours.py

    python src/campbell_neighbours.py --token whyte --occurrence 1

    python src/campbell_neighbours.py --token flawme

    python src/campbell_neighbours.py --token fier

    python src/campbell_neighbours.py --token-idx 1462
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace


DEFAULT_DOC_ID = "Campbell-Apocalypse-Export-05-English-Prose-tei"
DEFAULT_CORPUS = "misc"

DEFAULT_TOKEN = "whyte"
DEFAULT_OCCURRENCE = 2

DEFAULT_START_YEAR = 1000
DEFAULT_END_YEAR = 1950

DEFAULT_SCALE = "medium"
DEFAULT_PER_BUCKET_K = 50
DEFAULT_TOP_N = 50


@dataclass(frozen=True)
class Event:
    event_id: int
    corpus: str
    doc_id: str
    token_idx: int
    token: str
    pub_year: int
    title: str | None
    author: str | None


def find_campbell_event(
    *,
    corpus: str,
    doc_id: str,
    token: str,
    occurrence: int,
) -> Event:
    """
    Find the nth occurrence of a literal token in the specified document.

    Ordering is by token_idx, then event_id. This makes repeated forms,
    such as the two "whyte" tokens in the Campbell passage,
    unambiguous.
    """
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1")

    with get_connection(application_name="campbell-neighbours") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.corpus,
                    e.doc_id,
                    e.token_idx,
                    e.token,
                    e.pub_year,
                    d.title,
                    d.author
                FROM events AS e
                JOIN documents AS d
                  ON d.doc_id = e.doc_id
                WHERE e.corpus = %s
                  AND e.doc_id = %s
                  AND e.token = %s
                ORDER BY e.token_idx, e.event_id
                """,
                (corpus, doc_id, token),
            )

            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No event found for corpus={corpus!r}, "
            f"doc_id={doc_id!r}, token={token!r}"
        )

    if occurrence > len(rows):
        raise RuntimeError(
            f"Requested occurrence {occurrence} of token {token!r}, "
            f"but only {len(rows)} occurrence(s) were found in "
            f"{doc_id!r}."
        )

    row = rows[occurrence - 1]

    return Event(
        event_id=row[0],
        corpus=row[1],
        doc_id=row[2],
        token_idx=row[3],
        token=row[4],
        pub_year=row[5],
        title=row[6],
        author=row[7],
    )


def find_campbell_event_by_token_idx(
    *,
    corpus: str,
    doc_id: str,
    token_idx: int,
) -> Event:
    """Find an event by exact document/token position."""
    with get_connection(application_name="campbell-neighbours") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.corpus,
                    e.doc_id,
                    e.token_idx,
                    e.token,
                    e.pub_year,
                    d.title,
                    d.author
                FROM events AS e
                JOIN documents AS d
                  ON d.doc_id = e.doc_id
                WHERE e.corpus = %s
                  AND e.doc_id = %s
                  AND e.token_idx = %s
                ORDER BY e.event_id
                """,
                (corpus, doc_id, token_idx),
            )

            rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No event found for corpus={corpus!r}, "
            f"doc_id={doc_id!r}, token_idx={token_idx}"
        )

    if len(rows) > 1:
        raise RuntimeError(
            f"Multiple events found for token_idx={token_idx}: "
            f"{[row[0] for row in rows]}"
        )

    row = rows[0]

    return Event(
        event_id=row[0],
        corpus=row[1],
        doc_id=row[2],
        token_idx=row[3],
        token=row[4],
        pub_year=row[5],
        title=row[6],
        author=row[7],
    )


def get_context(
    conn,
    *,
    doc_id: str,
    token_idx: int,
    radius: int = 20,
) -> str:
    """Return a small PostgreSQL token context around an event."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_idx, token
            FROM events
            WHERE doc_id = %s
              AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx
            """,
            (
                doc_id,
                token_idx - radius,
                token_idx + radius,
            ),
        )

        rows = cur.fetchall()

    return " ".join(row[1] for row in rows)


def fetch_metadata(event_ids: list[int]) -> dict[int, dict]:
    """Resolve corpus metadata for candidate event IDs."""
    if not event_ids:
        return {}

    with get_connection(application_name="campbell-neighbours") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    e.event_id,
                    e.corpus,
                    e.doc_id,
                    e.token_idx,
                    e.token,
                    e.pub_year,
                    d.title,
                    d.author
                FROM events AS e
                JOIN documents AS d
                  ON d.doc_id = e.doc_id
                WHERE e.event_id = ANY(%s)
                """,
                (event_ids,),
            )

            rows = cur.fetchall()

    result = {}

    for row in rows:
        result[row[0]] = {
            "event_id": row[0],
            "corpus": row[1],
            "doc_id": row[2],
            "token_idx": row[3],
            "token": row[4],
            "pub_year": row[5],
            "title": row[6],
            "author": row[7],
        }

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find corpus-wide MacBERTh semantic neighbours of an "
            "actual Campbell Apocalypse observation."
        )
    )

    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"Campbell corpus (default: {DEFAULT_CORPUS})",
    )

    parser.add_argument(
        "--doc-id",
        default=DEFAULT_DOC_ID,
        help=f"Campbell document ID (default: {DEFAULT_DOC_ID})",
    )

    parser.add_argument(
        "--token",
        default=DEFAULT_TOKEN,
        help=f"Literal token to use as the query (default: {DEFAULT_TOKEN!r})",
    )

    parser.add_argument(
        "--occurrence",
        type=int,
        default=DEFAULT_OCCURRENCE,
        help=(
            "1-based occurrence of --token within the document "
            f"(default: {DEFAULT_OCCURRENCE})"
        ),
    )

    parser.add_argument(
        "--token-idx",
        type=int,
        default=None,
        help=(
            "Use an exact events.token_idx instead of "
            "--token/--occurrence."
        ),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help=f"First search year (default: {DEFAULT_START_YEAR})",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help=f"Last search year (default: {DEFAULT_END_YEAR})",
    )

    parser.add_argument(
        "--scale",
        default=DEFAULT_SCALE,
        choices=("local", "medium", "broad"),
        help=f"MacBERTh observation scale (default: {DEFAULT_SCALE})",
    )

    parser.add_argument(
        "--per-bucket-k",
        type=int,
        default=DEFAULT_PER_BUCKET_K,
        help=(
            "Number of Lance candidates retrieved from each chronological "
            f"bucket (default: {DEFAULT_PER_BUCKET_K})"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=(
            f"Number of final neighbours to display "
            f"(default: {DEFAULT_TOP_N})"
        ),
    )

    parser.add_argument(
        "--include-campbell",
        action="store_true",
        help="Include Campbell observations in the final results.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Locate the actual Campbell event.
    # ------------------------------------------------------------------

    if args.token_idx is not None:
        event = find_campbell_event_by_token_idx(
            corpus=args.corpus,
            doc_id=args.doc_id,
            token_idx=args.token_idx,
        )
    else:
        event = find_campbell_event(
            corpus=args.corpus,
            doc_id=args.doc_id,
            token=args.token,
            occurrence=args.occurrence,
        )

    print(f"Campbell document:   {event.doc_id}")
    print(f"Campbell token:      {event.token!r}")

    if args.token_idx is None:
        print(f"Campbell occurrence: {args.occurrence}")
    else:
        print("Campbell occurrence: n/a")

    print(f"Campbell token index: {event.token_idx}")
    print(f"Search years:        {args.start_year}–{args.end_year}")
    print(f"Scale:               {args.scale}")

    print()
    print("Query event:")
    print(f"  event_id   = {event.event_id}")
    print(f"  pub_year   = {event.pub_year}")
    print(f"  token_idx  = {event.token_idx}")
    print(f"  token      = {event.token!r}")
    print(f"  corpus     = {event.corpus}")
    print(f"  title      = {event.title!r}")
    print(f"  author     = {event.author!r}")

    # ------------------------------------------------------------------
    # Reconstruct the actual stored Campbell vector from Lance.
    # ------------------------------------------------------------------

    store = LanceObservationIndexStore(LANCE_INDEXES_DIR)

    query_space = SearchSpace(
        years=(event.pub_year, event.pub_year),
        scale=(args.scale,),
    )

    indexes = store.get(query_space)

    if args.scale not in indexes:
        raise RuntimeError(
            f"No Lance index available for scale {args.scale!r} "
            f"at publication year {event.pub_year}."
        )

    lance_index = indexes[args.scale]

    vectors = lance_index.reconstruct_many([event.event_id])

    if vectors.shape != (1, 768):
        raise RuntimeError(
            f"Expected one 768-dimensional vector for event "
            f"{event.event_id}, got shape={vectors.shape}"
        )

    query_vector = vectors[0]

    # Lance reconstruct_many() returns the stored vector. Normalize it
    # before using it as the cosine-search query.
    norm = np.linalg.norm(query_vector)

    if norm < 1e-12:
        raise RuntimeError(
            f"Reconstructed vector for event {event.event_id} "
            f"has near-zero norm."
        )

    query_vector = (query_vector / norm).astype(np.float32)

    print()
    print(
        f"Reconstructed vector: "
        f"shape={query_vector.shape}, dtype={query_vector.dtype}"
    )
    print(f"Vector norm: {np.linalg.norm(query_vector):.6f}")

    # ------------------------------------------------------------------
    # Search the entire chronological corpus.
    # ------------------------------------------------------------------

    search_space = SearchSpace(
        years=(args.start_year, args.end_year),
        scale=(args.scale,),
    )

    candidates: dict[int, float] = {}
    bucket_count = 0

    results = store.diachronic_search(
        {args.scale: query_vector},
        search_space,
        k=args.per_bucket_k,
    )

    for bucket, results_by_scale in results:
        bucket_count += 1

        search_result = results_by_scale[args.scale]

        for event_id, distance in zip(
            search_result.event_ids,
            search_result.distances,
        ):
            event_id = int(event_id)
            distance = float(distance)

            previous = candidates.get(event_id)

            if previous is None or distance < previous:
                candidates[event_id] = distance

    logger.info(
        "searched %d chronological buckets; "
        "%d unique candidate events",
        bucket_count,
        len(candidates),
    )

    # Never return the query event itself.
    candidates.pop(event.event_id, None)

    if not candidates:
        print("No candidate neighbours found.")
        return

    # ------------------------------------------------------------------
    # Resolve metadata.
    # ------------------------------------------------------------------

    ranked = sorted(
        candidates.items(),
        key=lambda item: item[1],
    )

    metadata = fetch_metadata(
        [event_id for event_id, _ in ranked]
    )

    # Filter before assigning display ranks. This prevents gaps such as
    # 5, 7, 13 when Campbell observations are excluded.
    display_candidates = []

    for event_id, distance in ranked:
        item = metadata.get(event_id)

        if item is None:
            continue

        if (
            not args.include_campbell
            and item["doc_id"] == event.doc_id
        ):
            continue

        display_candidates.append((event_id, distance, item))

        if len(display_candidates) >= args.top_n:
            break

    if not display_candidates:
        print("No displayable neighbours found.")
        return

    # ------------------------------------------------------------------
    # Print results with PostgreSQL context.
    # ------------------------------------------------------------------

    print()
    print(
        f"Top {len(display_candidates)} corpus-wide neighbours:"
    )
    print()

    with get_connection(
        application_name="campbell-neighbours"
    ) as conn:
        for rank, (event_id, distance, item) in enumerate(
            display_candidates,
            start=1,
        ):
            context = get_context(
                conn,
                doc_id=item["doc_id"],
                token_idx=item["token_idx"],
            )

            print(
                f"{rank:>3}. "
                f"distance={distance:.6f} "
                f"year={item['pub_year']} "
                f"event={event_id}"
            )

            print(
                f"     {item['token']!r} "
                f"{item['corpus']} "
                f"{item['doc_id']}"
            )

            print(
                f"     {item['author'] or '[unknown author]'} — "
                f"{item['title'] or '[untitled]'}"
            )

            print(f"     ... {context} ...")
            print()


if __name__ == "__main__":
    main()