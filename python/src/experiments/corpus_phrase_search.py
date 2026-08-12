#!/usr/bin/env python
"""
corpus_phrase_search.py

Resolve a lexical phrase against the complete corpus and, where Tier 1
contains the corresponding observations, return their event identities.

The corpus is authoritative for lexical occurrence search.
Tier 1 is authoritative for contextual observation identity.
Tier 2 consumes those Tier 1 observations; it does not perform corpus search.

Important invariants:

- Phrase matching is performed against corpus token order.
- token_idx refers to the original corpus token position.
- Matching is case-insensitive.
- Punctuation and other corpus tokens remain part of token order.
- A phrase match identifies a corpus occurrence, not a Tier 2 concept.
- A corpus occurrence may correspond to multiple Tier 1 observation events.
- corpus is part of corpus-occurrence identity because doc_id is not globally
  unique across corpora.
- Missing Tier 1 observations are reported rather than silently discarded.
- Diagnostic inspection is read-only and does not repair missing observations.
"""

from __future__ import annotations

import argparse

from lib.corpus_config import EVENTSTORE_T1_PATH
from lib.corpus_db import get_connection
from lib.zarr_event_lookup import ZarrEventLookup


def normalise_phrase(phrase: str) -> list[str]:
    """
    Convert the user query into the exact lexical tokens used for matching.
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
    Find exact consecutive token sequences in the corpus.

    Matching is performed on token_idx rather than textual reconstruction so
    corpus tokenisation remains authoritative.
    """
    aliases = [f"t{i}" for i in range(len(phrase))]

    joins = []
    where = []
    params: list[object] = []

    for i, token in enumerate(phrase):
        alias = aliases[i]

        if i > 0:
            previous = aliases[i - 1]

            joins.append(
                f"""
                JOIN pamphlet_tokens {alias}
                  ON {alias}.corpus = {previous}.corpus
                 AND {alias}.doc_id = {previous}.doc_id
                 AND {alias}.token_idx = {previous}.token_idx + 1
                """
            )

        where.append(f"LOWER({alias}.token) = %s")
        params.append(token)

    if corpus is not None:
        where.append("t0.corpus = %s")
        params.append(corpus)

    query = f"""
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

    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def resolve_tier1_events(lookup, occurrences):
    """
    Resolve corpus occurrences to Tier 1 observation event IDs.

    A corpus occurrence may have multiple Tier 1 observations because the
    same token can be represented under multiple contextual windows.

    Missing Tier 1 observations are reported rather than discarded.
    """
    event_ids_by_position = lookup.find_event_ids_by_positions(
        occurrences
    )

    resolved = []
    missing = []

    for corpus, doc_id, token_idx in occurrences:
        key = (
            str(corpus),
            str(doc_id),
            int(token_idx),
        )

        event_ids = event_ids_by_position.get(key, [])

        if event_ids:
            resolved.append(
                {
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                    "event_ids": event_ids,
                }
            )
        else:
            missing.append(
                {
                    "corpus": corpus,
                    "doc_id": doc_id,
                    "token_idx": token_idx,
                }
            )

    return resolved, missing


def inspect_missing_occurrences(
    conn,
    lookup,
    missing,
    radius: int = 5,
):
    """
    Inspect corpus/Tier 1 alignment for every unresolved occurrence.

    This is diagnostic only. It does not modify corpus or Tier 1 data.

    The diagnostic retrieves a small corpus-token window around each missing
    occurrence and reports any Tier 1 events found at those same coordinates.
    """
    diagnostics = []

    for occurrence in missing:
        corpus = occurrence["corpus"]
        doc_id = occurrence["doc_id"]
        token_idx = int(occurrence["token_idx"])

        start_idx = max(0, token_idx - radius)
        end_idx = token_idx + radius

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    corpus,
                    doc_id,
                    token_idx,
                    token
                FROM pamphlet_tokens
                WHERE corpus = %s
                  AND doc_id = %s
                  AND token_idx BETWEEN %s AND %s
                ORDER BY token_idx
                """,
                (
                    corpus,
                    doc_id,
                    start_idx,
                    end_idx,
                ),
            )

            corpus_rows = cur.fetchall()

        positions = [
            (
                row[0],
                row[1],
                row[2],
            )
            for row in corpus_rows
        ]

        event_ids_by_position = lookup.find_event_ids_by_positions(
            positions
        )

        tokens = []

        for row in corpus_rows:
            row_corpus, row_doc_id, row_token_idx, token = row

            key = (
                str(row_corpus),
                str(row_doc_id),
                int(row_token_idx),
            )

            tokens.append(
                {
                    "token_idx": int(row_token_idx),
                    "token": str(token),
                    "event_ids": event_ids_by_position.get(
                        key,
                        [],
                    ),
                }
            )

        diagnostics.append(
            {
                "corpus": corpus,
                "doc_id": doc_id,
                "token_idx": token_idx,
                "tokens": tokens,
            }
        )

    return diagnostics


def search_phrase(
    phrase: str,
    *,
    corpus: str | None = None,
    lookup=None,
):
    """
    Resolve a corpus phrase to Tier 1 event IDs.

    This is the reusable API used by tests, CLI tools and eventually the GUI.

    The returned counts deliberately distinguish corpus occurrences,
    matched corpus occurrences, and Tier 1 observation events.
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
        lookup = ZarrEventLookup(EVENTSTORE_T1_PATH)

    resolved, missing = resolve_tier1_events(
        lookup,
        occurrences,
    )

    tier1_event_count = sum(
        len(occurrence["event_ids"])
        for occurrence in resolved
    )

    return {
        "phrase": phrase,
        "tokens": tokens,
        "corpus": corpus,
        "occurrences": len(occurrences),
        "matched": len(resolved),
        "missing": len(missing),
        "tier1_events": tier1_event_count,
        "matches": resolved,
        "missing_occurrences": missing,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase")
    parser.add_argument("--corpus", default=None)
    parser.add_argument(
        "--diagnostic-radius",
        type=int,
        default=5,
        help="Number of corpus tokens to show on each side of missing occurrences",
    )
    args = parser.parse_args()

    lookup = ZarrEventLookup(EVENTSTORE_T1_PATH)

    result = search_phrase(
        args.phrase,
        corpus=args.corpus,
        lookup=lookup,
    )

    print(f"phrase: {result['phrase']}")
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

    if not result["missing_occurrences"]:
        return

    print("\nMissing from Tier 1:")

    for occurrence in result["missing_occurrences"]:
        print(
            occurrence["corpus"],
            occurrence["doc_id"],
            occurrence["token_idx"],
        )

    conn = get_connection()

    try:
        diagnostics = inspect_missing_occurrences(
            conn,
            lookup,
            result["missing_occurrences"],
            radius=args.diagnostic_radius,
        )
    finally:
        conn.close()

    print("\nDiagnostics:")

    for diagnostic in diagnostics:
        print(
            f"\n{diagnostic['corpus']} "
            f"{diagnostic['doc_id']} "
            f"{diagnostic['token_idx']}"
        )

        for row in diagnostic["tokens"]:
            marker = (
                " <-- TARGET"
                if row["token_idx"] == diagnostic["token_idx"]
                else ""
            )

            print(
                f"  {row['token_idx']:>6} "
                f"{row['token']!r:<25} "
                f"Tier1={row['event_ids']}"
                f"{marker}"
            )


if __name__ == "__main__":
    main()
