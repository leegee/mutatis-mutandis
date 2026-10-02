#!/usr/bin/env python

"""
test_phrase_search6.py

Compare MacBERTh retrieval for WHITE and GREY hair probes.

Each probe is encoded directly as complete text using
MacBertMeanPhraseEncoder.encode_text(). No carrier sentence is used.

Lance provides the nearest observations, independently within each
chronological bucket. PostgreSQL provides authoritative event/document
provenance and source-token context.

Results are reported by chronological bucket rather than merged into one
global ranking.

No PostgreSQL data is modified.
"""

from __future__ import annotations

import sys
import argparse
from dataclasses import dataclass, field

import numpy as np

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCALE = "medium"

PHRASE_PROBES: dict[str, list[str]] = {
    "WHITE": [
        "he had white hair",
        "his hair was white",
        "hair white as wool",
    ],
    "GREY": [
        "he had grey hair",
        "his hair was grey",
        "hair grey as wool",
    ],
}

MIN_YEAR = 1500
MAX_YEAR = 1949

# Number retrieved from each chronological bucket, per probe.
TOP_N = 10

# Maximum number displayed from each bucket after global event deduplication.
DISPLAY_LIMIT = 30

# Number of tokens on either side of the event token in the quoted context.
CONTEXT_TOKENS = 256


# ---------------------------------------------------------------------------
# Retrieved observation
# ---------------------------------------------------------------------------

@dataclass
class RetrievedObservation:
    event_id: int
    event: dict

    # Each retrieval retains:
    #
    #   bucket
    #   concept
    #   phrase
    #   distance
    #
    retrievals: list[
        tuple[tuple[int, int], str, str, float]
    ] = field(default_factory=list)

    @property
    def white_distance(self) -> float | None:
        distances = [
            distance
            for _bucket, concept, _phrase, distance
            in self.retrievals
            if concept == "WHITE"
        ]

        return min(distances) if distances else None

    @property
    def grey_distance(self) -> float | None:
        distances = [
            distance
            for _bucket, concept, _phrase, distance
            in self.retrievals
            if concept == "GREY"
        ]

        return min(distances) if distances else None

    @property
    def difference(self) -> float | None:
        """
        WHITE distance - GREY distance.

        Negative -> WHITE is closer.
        Positive -> GREY is closer.
        Zero     -> equal distance.
        """

        white = self.white_distance
        grey = self.grey_distance

        if white is None or grey is None:
            return None

        return white - grey


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def fetch_event_metadata(
    connection,
    event_ids: list[int],
) -> dict[int, dict]:
    """
    Fetch authoritative event provenance and document metadata.

    Events provide token-level provenance.
    Documents provides bibliographic metadata.
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

    metadata: dict[int, dict] = {}

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
    Fetch source tokens around an event.
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
    Render the source context and mark the retrieved token.
    """

    rendered = []

    for index, token in context:

        if index == token_idx:
            rendered.append(f"[{token}]")
        else:
            rendered.append(token)

    return " ".join(rendered)


def quote_context(
    context: list[tuple[int, str]],
    token_idx: int,
    *,
    max_tokens: int = 120,
) -> str:
    """
    Produce a compact quoted passage around the retrieved token.

    The full context is still fetched, but only a bounded number of tokens
    are printed so that the output remains usable.
    """

    if not context:
        return "[no context found]"

    event_position = None

    for position, (index, _token) in enumerate(context):
        if index == token_idx:
            event_position = position
            break

    if event_position is None:
        selected = context[:max_tokens]

    else:
        half = max_tokens // 2

        start = max(
            0,
            event_position - half,
        )

        end = min(
            len(context),
            start + max_tokens,
        )

        # If we reached the end, move the start backwards where possible.
        start = max(
            0,
            end - max_tokens,
        )

        selected = context[start:end]

    rendered = []

    for index, token in selected:

        if index == token_idx:
            rendered.append(
                f"[{token}]"
            )
        else:
            rendered.append(token)

    prefix = "… " if selected[0][0] > context[0][0] else ""
    suffix = " …" if selected[-1][0] < context[-1][0] else ""

    return (
        '"'
        + prefix
        + " ".join(rendered)
        + suffix
        + '"'
    )


# ---------------------------------------------------------------------------
# Lance retrieval
# ---------------------------------------------------------------------------

def search_phrase(
    store: LanceObservationIndexStore,
    encoder: MacBertMeanPhraseEncoder,
    phrase: str,
    *,
    min_year: int,
    max_year: int,
    top_n: int,
) -> list[
    tuple[tuple[int, int], int, float]
]:
    """
    Search one phrase.

    Lance searches each chronological bucket independently.

    Returns:
        [
            ((bucket_start, bucket_end), event_id, distance),
            ...
        ]

    There is deliberately no global ranking here.
    """

    logger.info(
        "Searching probe %r",
        phrase,
    )

    query_vector = encoder.encode_text(
        phrase
    )

    if query_vector.shape != (768,):
        raise ValueError(
            f"Unexpected query vector shape for {phrase!r}: "
            f"{query_vector.shape}"
        )

    queries_by_scale = {
        SCALE: query_vector,
    }

    search_space = SearchSpace(
        years=(min_year, max_year),
        scale=(SCALE,),
    )

    results = []

    for (
        bucket_start,
        bucket_end,
    ), results_by_scale in store.diachronic_search(
        queries_by_scale,
        search_space,
        k=top_n,
    ):

        result = results_by_scale[SCALE]

        for event_id, distance in zip(
            result.event_ids,
            result.distances,
        ):

            event_id = int(event_id)
            distance = float(distance)

            if not np.isfinite(distance):
                continue

            results.append(
                (
                    (bucket_start, bucket_end),
                    event_id,
                    distance,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_observations(
    retrieved: dict[int, RetrievedObservation],
    *,
    concept: str,
    phrase: str,
    hits: list[
        tuple[tuple[int, int], int, float]
    ],
    connection,
) -> None:
    """
    Merge one probe's bucketed Lance results.

    Bucket information is deliberately retained.
    """

    event_ids = list(
        dict.fromkeys(
            event_id
            for _bucket, event_id, _distance
            in hits
        )
    )

    metadata = fetch_event_metadata(
        connection,
        event_ids,
    )

    for bucket, event_id, distance in hits:

        event = metadata[event_id]

        observation = retrieved.get(
            event_id
        )

        if observation is None:

            observation = RetrievedObservation(
                event_id=event_id,
                event=event,
            )

            retrieved[event_id] = observation

        observation.retrievals.append(
            (
                bucket,
                concept,
                phrase,
                distance,
            )
        )


# ---------------------------------------------------------------------------
# Bucket output
# ---------------------------------------------------------------------------

def print_bucket_results(
    observations: dict[int, RetrievedObservation],
    connection,
    *,
    display_limit_per_bucket: int,
) -> None:
    """
    Print observations chronologically by retrieval bucket.

    Results are ranked by distance within each bucket.

    Each result includes an actual quoted passage from the corpus.
    """

    # --------------------------------------------------------------
    # Find the best bucket/distance for each event.
    # This performs global event deduplication.
    # --------------------------------------------------------------

    event_bucket_distance: dict[
        int,
        tuple[tuple[int, int], float],
    ] = {}

    for observation in observations.values():

        for (
            bucket,
            _concept,
            _phrase,
            distance,
        ) in observation.retrievals:

            previous = event_bucket_distance.get(
                observation.event_id
            )

            if (
                previous is None
                or distance < previous[1]
            ):

                event_bucket_distance[
                    observation.event_id
                ] = (
                    bucket,
                    distance,
                )

    # --------------------------------------------------------------
    # Assign unique observations to their retained bucket.
    # --------------------------------------------------------------

    bucket_events: dict[
        tuple[int, int],
        list[RetrievedObservation],
    ] = {}

    for event_id, (
        bucket,
        _distance,
    ) in event_bucket_distance.items():

        bucket_events.setdefault(
            bucket,
            [],
        ).append(
            observations[event_id]
        )

    # --------------------------------------------------------------
    # Print chronologically.
    # --------------------------------------------------------------

    for bucket in sorted(bucket_events):

        bucket_start, bucket_end = bucket

        bucket_observations = bucket_events[
            bucket
        ]

        def bucket_distance(
            observation: RetrievedObservation,
        ) -> float:

            distances = [
                distance
                for (
                    retrieval_bucket,
                    _concept,
                    _phrase,
                    distance,
                ) in observation.retrievals
                if retrieval_bucket == bucket
            ]

            return min(distances)

        bucket_observations.sort(
            key=bucket_distance
        )

        bucket_observations = bucket_observations[
            :display_limit_per_bucket
        ]

        print()
        print("=" * 100)
        print(
            f"{bucket_start}–{bucket_end}"
        )
        print("=" * 100)

        for rank, observation in enumerate(
            bucket_observations,
            start=1,
        ):

            event = observation.event
            distance = bucket_distance(
                observation
            )

            white = observation.white_distance
            grey = observation.grey_distance
            difference = observation.difference

            print()
            print(
                f"{rank:2d}. "
                f"distance={distance:.6f} "
                f"year={event['pub_year']} "
                f"event={observation.event_id}"
            )

            print(
                f"    author={event['author']!r}"
            )

            print(
                f"    title={event['title']!r}"
            )

            print(
                f"    corpus={event['corpus']} "
                f"doc={event['doc_id']} "
                f"token_idx={event['token_idx']}"
            )

            print(
                f"    WHITE="
                f"{white if white is not None else '—'}  "
                f"GREY="
                f"{grey if grey is not None else '—'}  "
                f"Δ="
                f"{difference if difference is not None else '—'}"
            )

            print()
            print(
                "    retrieved by:"
            )

            for (
                retrieval_bucket,
                concept,
                phrase,
                retrieval_distance,
            ) in sorted(
                observation.retrievals,
                key=lambda item: item[3],
            ):

                print(
                    f"      "
                    f"{retrieval_bucket[0]}–"
                    f"{retrieval_bucket[1]}  "
                    f"{concept:5s} "
                    f"{retrieval_distance:.6f}  "
                    f"{phrase!r}"
                )

            # ------------------------------------------------------
            # Actual corpus quotation.
            # ------------------------------------------------------

            context = fetch_context(
                connection,
                corpus=event["corpus"],
                doc_id=event["doc_id"],
                token_idx=event["token_idx"],
                radius=CONTEXT_TOKENS,
            )

            print()
            print(
                "    found:"
            )

            print(
                "    "
                + quote_context(
                    context,
                    event["token_idx"],
                )
            )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    observations: list[RetrievedObservation],
) -> None:

    white_count = sum(
        observation.white_distance is not None
        for observation in observations
    )

    grey_count = sum(
        observation.grey_distance is not None
        for observation in observations
    )

    both_count = sum(
        (
            observation.white_distance is not None
            and observation.grey_distance is not None
        )
        for observation in observations
    )

    print()
    print(
        f"unique observations:    {len(observations)}"
    )

    print(
        f"WHITE-retrieved:        {white_count}"
    )

    print(
        f"GREY-retrieved:         {grey_count}"
    )

    print(
        f"both WHITE and GREY:    {both_count}"
    )


# ---------------------------------------------------------------------------
# Probe resolution
# ---------------------------------------------------------------------------

def resolve_probe_phrases(
    *,
    phrase: str | None,
    concepts: list[str] | None,
    keys: list[str] | None,
) -> list[tuple[str | None, str]]:

    if phrase is not None:

        return [
            (None, phrase)
        ]

    if keys:

        result = []

        for key in keys:

            if key not in PHRASE_PROBES:

                raise ValueError(
                    f"Unknown probe key: {key!r}"
                )

            for item in PHRASE_PROBES[key]:

                result.append(
                    (key, item)
                )

        return result

    if concepts:

        result = []

        for concept in concepts:

            if concept not in PHRASE_PROBES:

                raise ValueError(
                    f"Unknown concept: {concept!r}"
                )

            for item in PHRASE_PROBES[concept]:

                result.append(
                    (concept, item)
                )

        return result

    return [
        (concept, phrase)
        for concept in PHRASE_PROBES
        for phrase in PHRASE_PROBES[concept]
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    sys.stdout.reconfigure(
        encoding="utf-8"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compare MacBERTh retrieval for WHITE and GREY "
            "hair probes."
        )
    )

    parser.add_argument(
        "--phrase",
        help="Search one explicit phrase.",
    )

    parser.add_argument(
        "--concept",
        action="append",
        choices=sorted(PHRASE_PROBES),
        help="Restrict retrieval to a probe concept.",
    )

    parser.add_argument(
        "--keys",
        nargs="+",
        choices=sorted(PHRASE_PROBES),
        help="Restrict retrieval to these probe groups.",
    )

    parser.add_argument(
        "--min-year",
        type=int,
        default=MIN_YEAR,
    )

    parser.add_argument(
        "--max-year",
        type=int,
        default=MAX_YEAR,
    )

    parser.add_argument(
        "--n",
        "--top",
        dest="top_n",
        type=int,
        default=TOP_N,
        help=(
            "Number of results retrieved per chronological bucket."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DISPLAY_LIMIT,
        help=(
            "Maximum number of unique observations displayed "
            "per chronological bucket."
        ),
    )

    parser.add_argument(
        "--lance-root",
        type=str,
        default=str(LANCE_INDEXES_DIR),
    )

    args = parser.parse_args()

    probes = resolve_probe_phrases(
        phrase=args.phrase,
        concepts=args.concept,
        keys=args.keys,
    )

    print(
        f"probes:                 {len(probes)}"
    )

    print(
        f"years:                  "
        f"{args.min_year}–{args.max_year}"
    )

    print(
        f"top per bucket:         {args.top_n}"
    )

    print(
        f"display limit/bucket:   {args.limit}"
    )

    print(
        f"scale:                  {SCALE}"
    )

    for concept, phrase in probes:

        if concept:

            print(
                f"probe:                  "
                f"{concept}: {phrase!r}"
            )

        else:

            print(
                f"probe:                  "
                f"{phrase!r}"
            )

    print()

    store = LanceObservationIndexStore(
        args.lance_root,
        available_years=range(
            args.min_year,
            args.max_year + 1,
        ),
        available_scales=(SCALE,),
    )

    encoder = MacBertMeanPhraseEncoder()

    retrieved: dict[
        int,
        RetrievedObservation,
    ] = {}

    with get_connection(
        application_name="semantic-phrase-probe",
    ) as connection:

        for concept, phrase in probes:

            effective_concept = (
                concept
                if concept is not None
                else "QUERY"
            )

            logger.info(
                "Searching %s probe %r",
                effective_concept,
                phrase,
            )

            hits = search_phrase(
                store=store,
                encoder=encoder,
                phrase=phrase,
                min_year=args.min_year,
                max_year=args.max_year,
                top_n=args.top_n,
            )

            merge_observations(
                retrieved,
                concept=effective_concept,
                phrase=phrase,
                hits=hits,
                connection=connection,
            )

        observations = list(
            retrieved.values()
        )

        print_summary(
            observations
        )

        print_bucket_results(
            retrieved,
            connection,
            display_limit_per_bucket=args.limit,
        )


if __name__ == "__main__":
    main()