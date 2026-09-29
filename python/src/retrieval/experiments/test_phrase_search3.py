#!/usr/bin/env python

"""
test_phrase_search3.py

Probe MacBERTh phrase semantics against the existing chronological Lance
observation indexes and display PostgreSQL token context around each hit.

A phrase is encoded independently of the corpus and used directly as a
query vector. Lance returns the nearest corpus observations; PostgreSQL
then supplies authoritative event provenance and source-token context.

The semantic probes can be used to find bodily-whiteness candidates, after
which the complete MacBERTh medium embedding window is cross-referenced
for eye vocabulary.

No PostgreSQL data is modified.

Phrase selection:

    probe.py "white hair"

Search one explicitly supplied phrase.

    probe.py "white hair,grey hair"

Search multiple explicitly supplied phrases.

    probe.py --concept WHITE

Search the predefined phrases associated with the WHITE concept set.

    probe.py --concept WHITE,BLACK

Search predefined phrases for multiple concept sets.

    probe.py

Search all predefined concept sets.

The canonical lexical forms and false positives remain defined in
corpus_config.py. This script does not duplicate them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lib.corpus_config import (
    CONCEPT_SETS,
    LANCE_INDEXES_DIR,
)
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


# ---------------------------------------------------------------------------
# Semantic phrase probes
# ---------------------------------------------------------------------------
#
# These are deliberately separate from CONCEPT_SETS.
#
# CONCEPT_SETS defines the canonical historical vocabulary and its
# false-positive rules.
#
# PHRASE_PROBES defines contextual semantic formulations that are encoded
# by MacBERTh and searched against the observation space.
#
# The keys must correspond to keys in CONCEPT_SETS.
#
# This means the lexical definition of WHITE remains in corpus_config.py,
# while these phrases define the semantic questions we want to probe.
#

PHRASE_PROBES: dict[str, list[str]] = {
    "WHITE": [
        "he had white hair",
        "hair white as wool",
        "luminous hair",
    ],
}


EYE_FORMS = frozenset(
    {
        "eye",
        "eyes",
        "eyne",
        "eie",
        "eien",
    }
)


def fetch_event_metadata(
    connection,
    event_ids: list[int],
) -> dict[int, dict]:
    """
    Fetch authoritative event provenance and source-document metadata.

    The medium-window provenance is required to recover the complete
    embedding window used by MacBERTh.
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
                e.medium_window_id,
                e.medium_window_token_pos,
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
            medium_window_id,
            medium_window_token_pos,
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
            "medium_window_id": (
                int(medium_window_id)
                if medium_window_id is not None
                else None
            ),
            "medium_window_token_pos": (
                int(medium_window_token_pos)
                if medium_window_token_pos is not None
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


def fetch_medium_window(
    connection,
    *,
    corpus: str,
    doc_id: str,
    window_id: int,
) -> list[tuple[int, str]]:
    """
    Fetch the complete source-token window represented by a medium
    MacBERTh observation.

    The window_id is the document-local source-token position at which
    the 512-token embedding window begins.
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
                window_id,
                window_id + 511,
            ),
        )

        return [
            (
                int(row[0]),
                str(row[1]),
            )
            for row in cursor.fetchall()
        ]


def find_eye_forms(
    window: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """
    Find historical eye forms in the complete embedding window.

    Matching is deliberately token-based rather than substring-based;
    this prevents forms such as 'eyebrow' from being counted as 'eye'.
    """

    return [
        (token_idx, token)
        for token_idx, token in window
        if token.casefold() in EYE_FORMS
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
    concept: str | None,
    store: LanceObservationIndexStore,
    connection,
    min_year: int,
    max_year: int,
    n: int,
    context_tokens: int,
) -> None:
    """
    Search one MacBERTh phrase independently in each physical Lance
    bucket and retain only observations whose complete medium embedding
    window contains an eye form.

    Results are deliberately not fused with RRF. The purpose is to inspect
    the raw semantic neighbourhood within each temporal population.
    """

    encoder = MacBertMeanPhraseEncoder()

    query_vector = encoder.encode_text( phrase )

    if query_vector.shape != (768,):
        raise RuntimeError(
            "Unexpected MacBERTh query-vector shape: "
            f"{query_vector.shape}"
        )

    logger.info(
        "[phrase-probe] concept=%r phrase=%r vector_shape=%s",
        concept,
        phrase,
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
        k=n,
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

        retained = []

        for event_id, distance in valid:

            event = metadata[event_id]

            window_id = event["medium_window_id"]

            if window_id is None:
                raise RuntimeError(
                    "Retrieved event has no medium-window provenance: "
                    f"event_id={event_id}"
                )

            medium_window = fetch_medium_window(
                connection,
                corpus=event["corpus"],
                doc_id=event["doc_id"],
                window_id=window_id,
            )

            eye_forms = find_eye_forms(
                medium_window,
            )

            if not eye_forms:
                continue

            retained.append(
                (
                    event_id,
                    distance,
                    event,
                    eye_forms,
                )
            )

        if not retained:
            logger.info(
                "[phrase-probe] no eye-bearing embedding windows for %d-%d",
                bucket_start,
                bucket_end,
            )
            continue

        print()
        print(
            f"{bucket_start}-{bucket_end}"
        )
        print("-" * 96)

        for rank, (
            event_id,
            distance,
            event,
            eye_forms,
        ) in enumerate(
            retained,
            start=1,
        ):

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

            eye_text = ", ".join(
                f"{token!r}@{token_idx}"
                for token_idx, token in eye_forms
            )

            print(
                f"\n"
                f"[{event['token']:6s}] "
                f"Rank {rank:3d} @ {distance:.6f} - "
                f"{event['corpus']:5} {event['pub_year']}"
            )

            print(
                f"     concept| {concept!r}"
            )
            print(
                f"     phrase | {phrase!r}"
            )
            print(
                f"     author | {event['author']!r}"
            )
            print(
                f"     title  | {event['title']!r}"
            )
            print(
                f"     eyes   | {eye_text}"
            )
            print(
                f"     text   | {context_text}"
            )


def resolve_probe_phrases(
    phrase_argument: str | None,
    concept_argument: str | None,
) -> list[tuple[str, str | None]]:
    """
    Resolve the phrases to search.

    An explicit positional phrase takes precedence over --concept.

    If no phrase is supplied, --concept selects entries from
    PHRASE_PROBES. Without --concept, all entries in PHRASE_PROBES
    are searched.

    Every PHRASE_PROBES key must correspond to a key in CONCEPT_SETS.
    """

    # Explicit CLI phrase: do not consult the probe tree.
    if phrase_argument is not None:

        phrases = [
            phrase.strip()
            for phrase in phrase_argument.split(",")
            if phrase.strip()
        ]

        return [
            (phrase, None)
            for phrase in phrases
        ]

    # Validate that our semantic probe groups refer to actual canonical
    # concept sets.
    unknown_probe_concepts = (
        set(PHRASE_PROBES) - set(CONCEPT_SETS)
    )

    if unknown_probe_concepts:
        raise RuntimeError(
            "PHRASE_PROBES contains concept(s) absent from CONCEPT_SETS: "
            f"{sorted(unknown_probe_concepts)}"
        )

    if concept_argument is None:

        concepts = list(PHRASE_PROBES)

    else:

        concepts = [
            concept.strip()
            for concept in concept_argument.split(",")
            if concept.strip()
        ]

    unknown_concepts = [
        concept
        for concept in concepts
        if concept not in CONCEPT_SETS
    ]

    if unknown_concepts:
        raise ValueError(
            "Unknown concept set(s): "
            f"{', '.join(unknown_concepts)}. "
            f"Available concept sets: {', '.join(CONCEPT_SETS)}"
        )

    missing_probes = [
        concept
        for concept in concepts
        if concept not in PHRASE_PROBES
    ]

    if missing_probes:
        raise ValueError(
            "No phrase probes are defined for concept set(s): "
            f"{', '.join(missing_probes)}"
        )

    return [
        (phrase, concept)
        for concept in concepts
        for phrase in PHRASE_PROBES[concept]
    ]


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Search chronological Lance observation indexes using "
            "MacBERTh phrase vectors and display source context."
        )
    )

    parser.add_argument(
        "phrase",
        nargs="?",
        default=None,
        help=(
            "Phrase to encode and search. Multiple phrases may be "
            "supplied as a comma-separated list. If omitted, use "
            "PHRASE_PROBES."
        ),
    )

    parser.add_argument(
        "--concept",
        "--keys",
        dest="concept",
        help=(
            "Comma-separated concept-set keys from PHRASE_PROBES "
            "to search when no positional phrase is supplied."
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
        "--n", "--top",
        type=int,
        default=TOP_N,
        help=(
            "Number of nearest-neighbour observations returned "
            "per chronological bucket."
        ),
    )

    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_TOKENS,
        help=(
            "Number of source tokens shown on either side of each "
            "matched event."
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

    if args.n <= 0:
        raise ValueError(
            "--n must be positive"
        )

    if args.context < 0:
        raise ValueError(
            "--context must not be negative"
        )

    probes = resolve_probe_phrases(
        args.phrase,
        args.concept,
    )

    if not probes:
        raise ValueError(
            "No phrases selected for probing."
        )

    logger.info(
        "[phrase-probe] Lance root=%s",
        args.lance_root,
    )

    logger.info(
        "[phrase-probe] selected %d probe phrase(s)",
        len(probes),
    )

    store = LanceObservationIndexStore(
        args.lance_root,
        available_years=range(
            args.min_year,
            args.max_year + 1,
        ),
        available_scales=(SCALE,),
    )

    with get_connection(
        application_name="semantic-phrase-probe",
    ) as connection:

        for phrase, concept in probes:

            print()

            if concept is not None:
                print(
                    f"=== {concept}: {phrase} ==="
                )
            else:
                print(
                    f"=== {phrase} ==="
                )

            search_phrase(
                phrase=phrase,
                concept=concept,
                store=store,
                connection=connection,
                min_year=args.min_year,
                max_year=args.max_year,
                n=args.n,
                context_tokens=args.context,
            )


if __name__ == "__main__":
    main()
