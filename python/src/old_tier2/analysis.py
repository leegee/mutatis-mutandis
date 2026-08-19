from __future__ import annotations

from collections import Counter

import numpy as np

from lib.corpus_logging import logger
from retrieval.diskann_multiscale import multiscale_diskann_search

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 2000
_NO_WPOS = -1


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def resolve_concept_positions(
    *,
    concept_name,
    concept,
    lookup,
    false_positives=None,
):
    """
    Resolve lexical seed events and group their dense positions by year.

    This retains positions only because the current Tier 2 orchestration
    needs them for the concept's complete year schedule. Retrieval itself
    remains batched.
    """
    forms = {
        f.lower()
        for f in concept.get("forms", [])
    }

    false_positives = {
        x.lower()
        for x in (false_positives or [])
    }

    logger.info(
        "[tier2] %s forms: %s",
        concept_name,
        sorted(forms)[:50],
    )

    event_ids = list(
        lookup.iter_matching_event_ids(
            forms,
            false_positives,
        )
    )

    logger.info(
        "[tier2] %s: %d events",
        concept_name,
        len(event_ids),
    )

    if not event_ids:
        return {
            "forms": forms,
            "false_positives": false_positives,
            "event_ids": [],
            "event_ids_set": set(),
            "positions": np.empty(0, dtype=np.int64),
            "by_year": {},
        }

    positions = np.asarray(
        [
            lookup.get_pos(event_id)
            for event_id in event_ids
        ],
        dtype=np.int64,
    )

    by_year = {}

    for pos in positions:
        year = int(lookup.pub_year[pos])
        by_year.setdefault(year, []).append(int(pos))

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "positions": positions,
        "by_year": by_year,
    }


def neighbour_search(
    *,
    lookup,
    positions,
    top_n,
    indexes,
    pub_year,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
):
    return multiscale_diskann_search(
        indexes=indexes,
        lookup=lookup,
        positions=np.asarray(
            positions,
            dtype=np.int64,
        ),
        top_n=top_n,
        pub_year=pub_year,
        rrf_k=rrf_k,
        oversample=oversample,
    )


def _build_batch_events(
    *,
    chunk_positions,
    fused,
    lookup,
    false_positives,
    token_counts,
    doc_counts,
    window_counts,
):
    batch_events = []

    for pos in chunk_positions:
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
            window_counts[(doc_id, window_id)] += 1

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "vector_id": int(lookup.vector_id[npos]),
                    "token": token,
                    "doc_id": doc_id,
                    "pub_year": int(lookup.pub_year[npos]),
                    "token_idx": int(lookup.token_idx[npos]),
                    "window_id": window_id,
                    "window_token_pos": (
                        None
                        if wpos == _NO_WPOS
                        else wpos
                    ),
                    "score": item["score"],
                    "score_local": item.get("score_local"),
                    "score_medium": item.get("score_medium"),
                    "score_broad": item.get("score_broad"),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        wpos = int(lookup.window_token_pos[pos])

        batch_events.append(
            {
                "event_id": event_id,
                "vector_id": int(lookup.vector_id[pos]),
                "token": str(lookup.token[pos]),
                "doc_id": str(lookup.doc_id[pos]),
                "pub_year": int(lookup.pub_year[pos]),
                "token_idx": int(lookup.token_idx[pos]),
                "window_id": int(lookup.window_id[pos]),
                "window_token_pos": (
                    None
                    if wpos == _NO_WPOS
                    else wpos
                ),
                "neighbours": neighbours_out,
            }
        )

    return batch_events


def iter_year_concept_batches(
    *,
    concept_name,
    concept,
    lookup,
    indexes,
    year,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
    resolved=None,
    batch_size=BATCH_SIZE,
    token_counts=None,
    doc_counts=None,
    window_counts=None,
):
    """
    Yield bounded event batches for one concept/year.

    Only one retrieval batch and its resulting event payload are resident
    at a time.
    """
    if resolved is None:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

    false_positives = resolved["false_positives"]
    event_ids_set = resolved["event_ids_set"]

    year_positions = resolved["by_year"].get(year, [])

    if not year_positions:
        return

    if token_counts is None:
        token_counts = Counter()

    if doc_counts is None:
        doc_counts = Counter()

    if window_counts is None:
        window_counts = Counter()

    for chunk_positions in _chunks(
        year_positions,
        batch_size,
    ):
        result = neighbour_search(
            lookup=lookup,
            positions=chunk_positions,
            top_n=top_n,
            indexes=indexes,
            pub_year=year,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        fused = {
            int(lookup.event_id[pos]): neighbours
            for pos, neighbours in zip(
                chunk_positions,
                result,
            )
        }

        yield {
            "type": "batch",
            "events": _build_batch_events(
                chunk_positions=chunk_positions,
                fused=fused,
                lookup=lookup,
                false_positives=false_positives,
                token_counts=token_counts,
                doc_counts=doc_counts,
                window_counts=window_counts,
            ),
            "seed_ids": event_ids_set,
        }
