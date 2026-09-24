#!/usr/bin/env python
"""
Probe MacBERTh phrase semantics against the existing chronological Lance
observation indexes and display PostgreSQL token context around each hit.

A phrase is encoded independently of the corpus and used directly as a
query vector. Lance returns the nearest corpus observations; PostgreSQL
then supplies authoritative event provenance and source-token context.

No PostgreSQL data is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.corpus_logging import logger

from retrieval.macberth_phrase_encoder2 import (
    MacBertMeanPhraseEncoder,
)
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace


SCALE = "medium"
MIN_YEAR = 1500
MAX_YEAR = 1949
TOP_N = 10
CONTEXT_TOKENS = 30


def fetch_event_metadata(
    connection,
    event_ids: list[int],
) -> dict[int, dict]:
    """
    Fetch authoritative event provenance and source-document metadata.

    Events remain the authoritative source for token-level provenance;
    documents supplies human-readable bibliographic metadata.
    """

    if not event_ids:
        return {}

    with connection.cursor() as cursor:

        cursor.execute(
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
            LEFT JOIN documents AS d
              ON d.corpus = e.corpus
             AND d.doc_id = e.doc_id
            WHERE e.event_id = ANY(%s)
            """,
            (event_ids,),
        )

        rows = cursor.fetchall()

    metadata = {}

    for row in rows:

        (
            event_id,
            corpus,
            doc_id,
            token_idx,
            token,
            pub_year,
            title,
            author,
        ) = row

        metadata[int(event_id)] = {
            "event_id": int(event_id),
            "corpus": str(corpus),
            "doc_id": str(doc_id),
            "token_idx": int(token_idx),
            "token": str(token),
            "pub_year": (
                int(pub_year)
                if pub_year is not None
                else None
            ),
            "title": (
                str(title)
                if title is not None
                else None
            ),
            "author": (
                str(author)
                if author is not None
                else None
            ),
        }

    missing = set(event_ids) - set(metadata)

    if missing:
        raise RuntimeError(
            "Lance returned event IDs absent from PostgreSQL: "
            f"{sorted(missing)[:10]}"
        )

    return metadata


def fetch_context(
    connection,
    *,
    corpus: str,
    doc_id: str,
    token_idx: int,
    radius: int,
) -> list[tuple[int, str]]:
    """
    Fetch the source token sequence around one event.

    Token position is the authoritative document-local coordinate, so the
    context remains correct even when event IDs are sparse or selective.
    """

    with connection.cursor() as cursor:

        cursor.execute(
            """
            SELECT
                token_idx,
                token
            FROM tokens
            WHERE corpus = %s
              AND doc_id = %s
              AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx
            """,
            (
                corpus,
                doc_id,
                max(0, token_idx - radius),
                token_idx + radius,
            ),
        )

        return [
            (
                int(row[0]),
                str(row[1]),
            )
            for row in cursor.fetchall()
        ]


def format_context(
    context: list[tuple[int, str]],
    token_idx: int,
) -> str:
    """
    Render a token window while marking the observed token.

    Token boundaries are preserved by joining with spaces. Punctuation is
    therefore visible rather than reconstructed into potentially altered
    source text.
    """

    rendered = []

    for index, token in context:

        if index == token_idx:
            rendered.append(
                f"[{token}]"
            )
        else:
            rendered.append(token)

    return " ".join(rendered)


def search_phrase(
    *,
    phrase: str,
    carrier: str,
    store: LanceObservationIndexStore,
    connection,
    min_year: int,
    max_year: int,
    top_n: int,
    context_tokens: int,
) -> None:
    """
    Search one MacBERTh phrase independently in each physical Lance
    bucket and display source context for each returned event.

    Results are deliberately not fused with RRF. The purpose is to inspect
    the raw semantic neighbourhood within each temporal population.
    """

    encoder = MacBertMeanPhraseEncoder()

    query_vector = encoder.encode(
        phrase,
        carrier,
    )

    if query_vector.shape != (768,):
        raise RuntimeError(
            "Unexpected MacBERTh query-vector shape: "
            f"{query_vector.shape}"
        )

    logger.info(
        "[phrase-probe] phrase=%r carrier=%r vector_shape=%s",
        phrase,
        carrier,
        query_vector.shape,
    )

    queries_by_scale = {
        SCALE: query_vector,
    }

    search_space = SearchSpace(
        years=(min_year, max_year),
        scale=(SCALE,),
    )

    for (
        bucket_start,
        bucket_end,
    ), results_by_scale in store.diachronic_search(
        queries_by_scale,
        search_space,
        k=top_n,
    ):

        result = results_by_scale[SCALE]

        event_ids = result.event_ids
        distances = result.distances

        valid = [
            (
                int(event_id),
                float(distance),
            )
            for event_id, distance in zip(
                event_ids,
                distances,
            )
            if int(event_id) >= 0
        ]

        if not valid:
            logger.info(
                "[phrase-probe] no results for %d-%d",
                bucket_start,
                bucket_end,
            )
            continue

        metadata = fetch_event_metadata(
            connection,
            [
                event_id
                for event_id, _ in valid
            ],
        )

        print()
        print(
            f"{bucket_start}-{bucket_end}"
        )
        print("-" * 96)

        for rank, (
            event_id,
            distance,
        ) in enumerate(
            valid,
            start=1,
        ):

            event = metadata[event_id]

            context = fetch_context(
                connection,
                corpus=event["corpus"],
                doc_id=event["doc_id"],
                token_idx=event["token_idx"],
                radius=context_tokens,
            )

            context_text = format_context(
                context,
                event["token_idx"],
            )

            print(
                f"{rank:3d}  "
                f"distance={distance:.6f}  "
                f"event={event_id}  "
                f"year={event['pub_year']}  "
                f"corpus={event['corpus']}  "
                f"doc={event['doc_id']}  "
                f"token_idx={event['token_idx']}  "
                f"token={event['token']!r}"
            )

            print(
                f"     author={event['author']!r}"
            )

            print(
                f"     title={event['title']!r}"
            )

            print(
                f"     {context_text}"
            )

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Search chronological Lance observation indexes using "
            "a MacBERTh phrase vector and display source context."
        )
    )

    parser.add_argument(
        "phrase",
        nargs="?",
        default="white hair",
        help="Phrase to encode and search.",
    )

    parser.add_argument(
        "--carrier",
        default="The person had {}.",
        help=(
            "Carrier sentence containing {} where the phrase is inserted."
        ),
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=MIN_YEAR,
        help="First publication year to search.",
    )

    parser.add_argument(
        "--max-year",
        type=int,
        default=MAX_YEAR,
        help="Last publication year to search.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=TOP_N,
        help=(
            "Number of nearest observations returned per "
            "chronological bucket."
        ),
    )

    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_TOKENS,
        help=(
            "Number of source tokens shown on either side "
            "of each matched event."
        ),
    )

    parser.add_argument(
        "--lance-root",
        type=Path,
        default=LANCE_INDEXES_DIR,
        help="Root directory containing the Lance indexes.",
    )

    args = parser.parse_args()

    if args.min_year > args.max_year:
        raise ValueError(
            "--min-year must not be greater than --max-year"
        )

    if args.top <= 0:
        raise ValueError(
            "--top must be positive"
        )

    if args.context < 0:
        raise ValueError(
            "--context must not be negative"
        )

    logger.info(
        "[phrase-probe] Lance root=%s",
        args.lance_root,
    )

    store = LanceObservationIndexStore(
        args.lance_root,
        available_years=range(
            args.min_year,
            args.max_year + 1,
        ),
        available_scales=(SCALE,),
    )

    phrases = [
        phrase.strip()
        for phrase in args.phrase.split(",")
        if phrase.strip()
    ]

    with get_connection(
        application_name="semantic-phrase-probe",
    ) as connection:
        for phrase in phrases:
            search_phrase(
                phrase=phrase,
                carrier=args.carrier,
                store=store,
                connection=connection,
                min_year=args.min_year,
                max_year=args.max_year,
                top_n=args.top,
                context_tokens=args.context,
            )

if __name__ == "__main__":
    main()
