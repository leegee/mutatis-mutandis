"""
tier2.analysis

Pure analysis: neighbourhood retrieval around lexical seeds.

No database writes, no index loading, no side effects beyond logging.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from lib.eebo_faiss import multiscale_search
from lib.corpus_logging import logger

K = 60
RRF_K = 60
OVERSAMPLE = 5
_NO_WPOS = -1

# Seed events processed (and written) per chunk. Bounds peak memory to
# roughly BATCH_SIZE * top_n neighbour dicts, regardless of how many
# events a concept matches in total. Frequent words (KING, LAW,
# PARLIAMENT, PEOPLE...) can match hundreds of thousands of events in
# an 8.5M-event corpus; without chunking, a single concept's in-memory
# payload can dwarf the whole rest of the run.
BATCH_SIZE = 2000


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
    The cheap, FAISS-free half of concept resolution: map a concept's
    lexical forms to event positions, grouped by publication year.

    Deliberately separated from analyse_concept so callers (e.g. an
    eviction scheduler) can find out which years a concept touches
    without paying for any FAISS search.
    """
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
            "forms": forms,
            "false_positives": false_positives,
            "event_ids": event_ids,
            "positions": None,
            "by_year": {},
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

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(int(e) for e in event_ids),
        "positions": positions,
        "by_year": by_year,
    }


def build_year_schedule(
    *,
    lookup,
    concepts_to_run,
    false_positives=None,
):
    """
    One cheap, FAISS-free pass over every concept in a batch.

    Returns only the small year sets needed to drive a year-major
    processing loop (outer = year, inner = concepts that touch it).
    Full position / event_id payloads are deliberately not retained;
    the caller re-resolves a concept when it actually processes that
    concept's slice of a year.

    Returns:
        years_by_concept: {concept_name: {year, ...}}
        concepts_by_year: {year: [(concept_name, concept), ...]}
            (stable order matching concepts_to_run)
    """
    years_by_concept = {}
    concepts_by_year = {}

    for concept_name, concept in concepts_to_run:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

        years = set(resolved["by_year"])
        years_by_concept[concept_name] = years

        for year in years:
            concepts_by_year.setdefault(year, []).append(
                (concept_name, concept)
            )

        # resolved goes out of scope; only the year set survives.

    return years_by_concept, concepts_by_year


# Backward-compatible alias used by older call sites / tests.
def build_eviction_schedule(
    *,
    lookup,
    concepts_to_run,
    false_positives=None,
):
    """
    Deprecated wrapper around build_year_schedule that also fabricates
    a last_use map (concept-major eviction). Prefer build_year_schedule
    + year-major processing so at most one year's indices are resident.
    """
    years_by_concept, concepts_by_year = build_year_schedule(
        lookup=lookup,
        concepts_to_run=concepts_to_run,
        false_positives=false_positives,
    )
    # last concept index that needs each year (concept-major order)
    name_to_idx = {
        name: i for i, (name, _) in enumerate(concepts_to_run)
    }
    last_use = {}
    for year, pairs in concepts_by_year.items():
        last_use[year] = max(name_to_idx[name] for name, _ in pairs)
    return years_by_concept, last_use


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
    evict_index_after_year=False,
    resolved=None,
):
    if resolved is None:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

    forms = resolved["forms"]
    false_positives = resolved["false_positives"]
    event_ids = resolved["event_ids"]
    positions = resolved["positions"]
    by_year = resolved["by_year"]

    if not event_ids:
        return {
            "concept": concept_name,
            "empty": True,
        }

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

        # Only useful when the caller can guarantee no other concept in
        # the same run will need this year again (e.g. a single-concept
        # run). Batch runs should instead use build_eviction_schedule +
        # explicit eviction in the caller's loop, since a given year is
        # commonly shared across many concepts.
        if evict_index_after_year and hasattr(indexes, "evict"):
            indexes.evict(year)

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
    """
    Turn one chunk's worth of seed positions + their already-fused
    neighbours into output event dicts, updating the running aggregate
    counters as it goes. Counters stay cheap (bounded by vocabulary /
    document cardinality) even though events don't.
    """
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

        batch_events.append(
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
    Yield bounded-size batches for *one concept, one publication year*.

    Intended for the year-major service loop: the caller holds at most
    one year's FAISS indices, walks every concept that touches that
    year, then evicts. Aggregate counters may be passed in so they
    accumulate across years for the same concept.

    Yields:
        {"type": "batch", "events": [...], "seed_ids": set(...)}
    or nothing if the concept has no events in this year.
    """
    if resolved is None:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

    false_positives = resolved["false_positives"]
    event_ids_set = resolved.get("event_ids_set") or set()
    year_positions = resolved["by_year"].get(year) or []

    if not year_positions:
        return

    if token_counts is None:
        token_counts = Counter()
    if doc_counts is None:
        doc_counts = Counter()
    if window_counts is None:
        window_counts = Counter()

    for chunk_positions in _chunks(year_positions, batch_size):
        chunk_arr = np.asarray(chunk_positions, dtype=np.int64)

        result = multiscale_search(
            indexes,
            lookup,
            chunk_arr,
            top_n,
            pub_year=year,
            rrf_k=rrf_k,
            oversample=oversample,
        )

        fused = {
            int(lookup.event_id[pos]): neighbours
            for pos, neighbours in zip(chunk_positions, result)
        }

        batch_events = _build_batch_events(
            chunk_positions=chunk_positions,
            fused=fused,
            lookup=lookup,
            false_positives=false_positives,
            token_counts=token_counts,
            doc_counts=doc_counts,
            window_counts=window_counts,
        )

        yield {
            "type": "batch",
            "events": batch_events,
            "seed_ids": event_ids_set,
        }


def iter_concept_batches(
    *,
    concept_name,
    concept,
    lookup,
    indexes,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    false_positives=None,
    resolved=None,
    batch_size=BATCH_SIZE,
    evict_after_years=None,
):
    """
    Streaming counterpart to analyse_concept (concept-major).

    Prefer the year-major path in orchestrator.service for multi-concept
    batches so at most one year's FAISS indices are ever resident.
    This generator remains for single-concept runs and tests.
    """
    if resolved is None:
        resolved = resolve_concept_positions(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            false_positives=false_positives,
        )

    forms = resolved["forms"]
    false_positives = resolved["false_positives"]
    event_ids = resolved["event_ids"]
    event_ids_set = resolved["event_ids_set"]
    by_year = resolved["by_year"]

    if not event_ids:
        yield {"type": "empty", "concept": concept_name}
        return

    evict_after_years = evict_after_years or set()

    token_counts = Counter()
    doc_counts = Counter()
    window_counts = Counter()

    for year, year_positions in by_year.items():
        for item in iter_year_concept_batches(
            concept_name=concept_name,
            concept=concept,
            lookup=lookup,
            indexes=indexes,
            year=year,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
            false_positives=false_positives,
            resolved=resolved,
            batch_size=batch_size,
            token_counts=token_counts,
            doc_counts=doc_counts,
            window_counts=window_counts,
        ):
            yield item

        if year in evict_after_years and hasattr(indexes, "evict"):
            indexes.evict(year)

    yield {
        "type": "final",
        "concept": concept_name,
        "forms": forms,
        "n_events": len(event_ids),
        "aggregate": {
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
    evict_index_after_year=False,
):
    """
    Heart of Tier 2: neighbourhood retrieval for every requested concept.

    Takes already-constructed resources and returns a pure result dict.
    No database writes, no index loading, no side effects beyond logging.

    Note: this still accumulates every concept's result in `output` before
    returning, so callers processing many concepts in one batch (e.g.
    tier2.orchestrator.service) should prefer analysing + writing one
    concept at a time instead of calling this over a large concept list.
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
            evict_index_after_year=evict_index_after_year,
        )

        if emit:
            emit("concept_done", {"concept": concept_name})

    logger.info("[tier2.run_tier2_core] Leave")
    return output
