"""
tier2.analysis

Pure analysis: neighbourhood retrieval around lexical seeds.

No database writes, no index loading, no side effects beyond logging.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from lib.eebo_faiss import multiscale_search
from lib.eebo_logging import logger

K = 60
RRF_K = 60
OVERSAMPLE = 5
_NO_WPOS = -1


def analyse_concept(
    *,
    concept_name,
    concept,
    lookup,
    indexes,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
):
    forms = {
        f.lower()
        for f in concept.get("forms", [])
    }

    false_positives = {
        x.lower()
        for x in (false_positives or [])
    }

    logger.info(f"[tier2] {concept_name} forms: {sorted(forms)[:50]}")

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    logger.info(f"[tier2] {concept_name}: {len(event_ids)} events")

    if not event_ids:
        return {
            "concept": concept_name,
            "empty": True,
        }

    positions = np.asarray(
        [
            lookup.get_pos(eid)
            for eid in event_ids
        ],
        dtype=np.int64,
    )

    # Group by publication year.
    #
    # This is deliberately based on Tier 1 Zarr metadata.
    # Document metadata may be incomplete or normalised differently.

    by_year = {}
    for pos in positions:
        year = int(lookup.pub_year[pos])

        by_year.setdefault(
            year,
            []
        ).append(pos)

    fused = {}
    for year, year_positions in by_year.items():
        result = multiscale_search(
            indexes,
            lookup,
            np.asarray(
                year_positions,
                dtype=np.int64,
            ),
            top_n,
            pub_year=year,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        for pos, neighbours in zip(
            year_positions,
            result,
        ):
            fused[int(lookup.event_id[pos])] = neighbours

    token_counts = Counter()
    doc_counts = Counter()
    window_counts = Counter()

    output_events = []
    for pos in positions:
        event_id = int(lookup.event_id[pos])

        neighbours_out = []
        for item in fused[event_id]:
            neighbour_id = item["event_id"]
            if neighbour_id == event_id:
                continue

            npos = lookup.get_pos(neighbour_id)
            token = str(lookup.token[npos])

            if token.lower() in false_positives:
                continue

            doc_id = str(lookup.doc_id[npos])
            window_id = int(lookup.window_id[npos])
            wpos = int(lookup.window_token_pos[npos])
            token_counts[token] += 1
            doc_counts[doc_id] += 1

            window_counts[
                (
                    doc_id,
                    window_id,
                )
            ] += 1

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "vector_id": int(lookup.vector_id[npos]),
                    "token": token,
                    "doc_id": doc_id,
                    "pub_year": int(lookup.pub_year[npos]),
                    "token_idx": int(lookup.token_idx[npos]),
                    "window_id": window_id,
                    "window_token_pos":
                        None
                        if wpos == _NO_WPOS
                        else wpos,
                    "score": item["rrf_score"],
                    "score_local": item.get("score_local"),
                    "score_medium": item.get("score_medium"),
                    "score_broad": item.get("score_broad"),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        output_events.append(
            {
                "event_id": event_id,
                "vector_id": int(lookup.vector_id[pos]),
                "token": str(lookup.token[pos]),
                "doc_id": str(lookup.doc_id[pos]),
                "pub_year": int(lookup.pub_year[pos]),
                "token_idx": int(lookup.token_idx[pos]),
                "window_id": int(lookup.window_id[pos]),
                "window_token_pos":
                    None
                    if int(
                        lookup.window_token_pos[pos]
                    ) == _NO_WPOS
                    else int(
                        lookup.window_token_pos[pos]
                    ),
                "neighbours": neighbours_out,
            }
        )

    return {
        "concept": concept_name,
        "forms": forms,
        "n_events": len(event_ids),
        "events": output_events,
        "aggregate":
            {
                "top_tokens": token_counts.most_common(top_n),
                "top_docs": doc_counts.most_common(top_n),
                "top_windows": window_counts.most_common(top_n),
            },
    }


def run_tier2_core(
    *,
    lookup,
    indexes,
    concepts_to_run,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    false_positives=None,
    emit=None,
):
    """
    Heart of Tier 2: neighbourhood retrieval for every requested concept.

    Takes already-constructed resources and returns a pure result dict.
    No database writes, no index loading, no side effects beyond logging.
    """
    logger.info("[tier2.run_tier2_core] Enter")
    output = {}

    for concept_name, concept in concepts_to_run:
        if emit:
            emit("concept_start", {"concept": concept_name})

        output[concept_name] = analyse_concept(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            indexes=indexes,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            false_positives=false_positives,
        )

        if emit:
            emit("concept_done", {"concept": concept_name})

    logger.info("[tier2.run_tier2_core] Leave")
    return output
