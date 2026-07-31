#!/usr/bin/env python
"""
corpus_phrase_search.py

Resolve an exact lexical phrase in the corpus and map each occurrence to
its Tier 1 contextual observation events.

The corpus is authoritative for lexical occurrence identity.
Tier 1 is authoritative for contextual observation identity.
Tier 2 consumes Tier 1 event IDs and does not perform corpus searching.

Important invariants:

- Phrase matching follows corpus token order.
- token_idx identifies the original corpus token position.
- corpus is part of corpus-occurrence identity.
- A corpus occurrence may resolve to multiple Tier 1 event IDs.
- Missing Tier 1 observations are reported, not silently discarded.
"""

from __future__ import annotations

import argparse

from lib.corpus_config import ZARR_PATH
from lib.corpus_db import get_connection
from lib.zarr_event_lookup import ZarrEventLookup


def normalise_phrase(phrase: str) -> list[str]:
    """
    Match the corpus token representation case-insensitively.
    """
    tokens = phrase.split()

    if not tokens:
        raise ValueError("Phrase must contain at least one token")

    return [token.lower() for token in tokens]


def find_phrase_occurrences(
    conn,
    phrase: list[str],
    corpus: str | None = None,
):
    """
    Find exact consecutive token sequences in pamphlet_tokens.

    Matching token_idx rather than reconstructed text keeps corpus
    tokenisation authoritative.
    """
    aliases = [f"t{i}" for i in range(len(phrase))]

    joins = []
    params: list[object] = [phrase[0]]

    for i in range(1, len(phrase)):
        previous = aliases[i - 1]
        current = aliases[i]

        joins.append(
            f"""
            JOIN pamphlet_tokens {current}
              ON {current}.corpus = {previous}.corpus
             AND {current}.doc_id = {previous}.doc_id
             AND {current}.token_idx = {previous}.token_idx + 1
            """
        )

        params.append(phrase[i])

    where = [
        "LOWER(t0.token) = %s",
    ]

    if corpus is not None:
        where.append("t0.corpus = %s")
        params.append(corpus)

    for i in range(1, len(phrase)):
        where.append(
            f"LOWER(t{i}.token) = %s"
        )

    sql = f"""
        SELECT
            t0.corpus,
            t0.doc_id,
            t0.token_idx
        FROM pamphlet_tokens t0
        {" ".join(joins)}
        WHERE {" AND ".join(where)}
        ORDER BY
            t0.corpus,
            t0.doc_id,
            t0.token_idx
    """

    # The WHERE placeholders are t0, optional corpus, then t1...tn.
    params = [phrase[0]]
    if corpus is not None:
        params.append(corpus)
    params.extend(phrase[1:])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def search_phrase(
    phrase: str,
    *,
    corpus: str | None = None,
    lookup=None,
):
    """
    Resolve a corpus phrase to Tier 1 event IDs.

    This is the reusable API used by tests, CLI tools and eventually the GUI.
    """
    tokens = normalise_phrase(phrase)

    conn = get_connection()

    try:
        occurrences = find_phrase_occurrences(
            conn,
            tokens,
            corpus=corpus,
        )
    finally:
        conn.close()

    if lookup is None:
        lookup = ZarrEventLookup(ZARR_PATH)

    positions = [
        (corpus, doc_id, token_idx)
        for corpus, doc_id, token_idx in occurrences
    ]

    event_ids_by_position = lookup.find_event_ids_by_positions(
        positions
    )

    resolved = []
    missing = []

    for occurrence_corpus, doc_id, token_idx in occurrences:
        key = (
            str(occurrence_corpus),
            str(doc_id),
            int(token_idx),
        )

        event_ids = event_ids_by_position.get(key, [])

        if event_ids:
            resolved.append(
                {
                    "corpus": occurrence_corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                    "event_ids": event_ids,
                }
            )
        else:
            missing.append(
                {
                    "corpus": occurrence_corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                }
            )

    return {
        "phrase": phrase,
        "tokens": tokens,
        "corpus": corpus,
        "occurrences": len(occurrences),
        "matched": len(resolved),
        "missing": len(missing),
        "tier1_events": sum(
            len(match["event_ids"])
            for match in resolved
        ),
        "matches": resolved,
        "missing_occurrences": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase")
    parser.add_argument("--corpus", default=None)
    args = parser.parse_args()

    result = search_phrase(
        args.phrase,
        corpus=args.corpus,
    )

    print(f"phrase: {result['phrase']}")
    print(f"corpus occurrences: {result['occurrences']}")
    print(f"Tier 1 matches: {result['matched']}")
    print(f"missing from Tier 1: {result['missing']}")
    print(f"corpus occurrences: {result['occurrences']}")
    print(f"Tier 1 occurrences: {result['matched']}")
    print(f"Tier 1 events: {result['tier1_events']}")
    print(f"missing from Tier 1: {result['missing']}")

    for match in result["matches"]:
        print(
            match["corpus"],
            match["doc_id"],
            match["token_idx"],
            match["event_ids"],
        )


if __name__ == "__main__":
    main()
