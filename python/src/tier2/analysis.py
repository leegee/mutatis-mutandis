"""
tier2/analysis.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 5

_NO_WPOS = -1



def _chunks(seq, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")

    for start in range(0, len(seq), size):
        yield seq[start:start + size]


def resolve_concept_positions(
    *,
    concept_name,
    concept,
    lookup,
    false_positives=None,
):
    forms = {
        str(form).lower()
        for form in concept.get("forms", [])
    }

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    logger.info(
        "[tier2] %s forms: %s",
        concept_name,
        sorted(forms)[:50],
    )

    event_ids = lookup.find_matching_event_ids(
        forms,
        false_positives,
    )

    event_ids = [int(event_id) for event_id in event_ids]

    logger.info(
        "[tier2] %s: %d seed events",
        concept_name,
        len(event_ids),
    )

    by_year = {}

    for event_id in event_ids:
        metadata = lookup.get_event_metadata(event_id)
        year = int(metadata["pub_year"])
        by_year.setdefault(year, []).append(event_id)

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "by_year": by_year,
    }


def _lance_where(
    *,
    year_start,
    year_end,
    model=None,
):
    clauses = []

    if year_start is not None:
        clauses.append(f"year >= {int(year_start)}")

    if year_end is not None:
        clauses.append(f"year <= {int(year_end)}")

    if model is not None:
        escaped = model.replace("'", "''")
        clauses.append(
            f"embedding_model = '{escaped}'"
        )

    return " AND ".join(clauses) if clauses else None


def _search_lance_table(
    *,
    table,
    queries,
    year_start,
    year_end,
    k,
    nprobes,
    executor,
):
    """
    Search one Lance scale table.

    Only event identity and distance cross the Lance/Python boundary.

    `executor` is supplied by the caller (multiscale_search) and shared
    across all scales/batches rather than recreated per call, since
    spinning up a fresh ThreadPoolExecutor for every few-row batch adds
    real overhead at BATCH_SIZE-sized granularity.
    """
    queries = np.asarray(
        queries,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError(
            "queries must be two-dimensional"
        )

    where = _lance_where(
        year_start=year_start,
        year_end=year_end,
    )

    def search_one(query):
        search = (
            table
            .search(query)
            .nprobes(nprobes)
        )

        if where:
            search = search.where(where)

        return (
            search
            .limit(k)
            .select([
                "event_id",
                "_distance",
            ])
            .to_list()
        )

    return list(executor.map(search_one, queries))


def multiscale_search(
    *,
    indexes,
    queries_by_scale,
    year_start,
    year_end,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    scales=SCALES,
    nprobes=20,
    executor=None,
):
    """
    Search each scale using its corresponding scale-specific embeddings.

    RRF merges scale rankings by stable event_id.

    `executor` lets a caller that runs multiscale_search many times (e.g.
    once per batch in iter_concept_batches) supply one long-lived
    ThreadPoolExecutor instead of paying setup/teardown cost per call. If
    omitted, a temporary executor is created and shut down before return,
    preserving the previous standalone behaviour.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    search_k = top_n * oversample
    results_by_scale = {}

    owns_executor = executor is None
    if owns_executor:
        executor = ThreadPoolExecutor(max_workers=2)

    try:
        for scale in scales:
            table = indexes.get(scale)

            if table is None:
                raise RuntimeError( f"Missing Lance table for scale={scale}" )

            queries = np.asarray(
                queries_by_scale[scale],
                dtype=np.float32,
            )

            if queries.ndim != 2:
                raise ValueError( f"queries for scale={scale} must be two-dimensional" )

            results_by_scale[scale] = _search_lance_table(
                table=table,
                queries=queries,
                year_start=year_start,
                year_end=year_end,
                k=search_k,
                nprobes=nprobes,
                executor=executor,
            )
    finally:
        if owns_executor:
            executor.shutdown(wait=True)

    first_scale = scales[0]
    query_count = len(
        queries_by_scale[first_scale]
    )

    for scale in scales[1:]:
        if len(queries_by_scale[scale]) != query_count:
            raise ValueError( "all scales must contain the same number of query vectors" )

    output = []

    for query_idx in range(query_count):
        fused = {}

        for scale in scales:
            result = results_by_scale[scale][query_idx]

            for rank, row in enumerate(
                result,
                start=1,
            ):
                event_id = int(row["event_id"])

                item = fused.setdefault(
                    event_id,
                    {
                        "score": 0.0,
                        "score_local": None,
                        "score_medium": None,
                        "score_broad": None,
                    },
                )

                item["score"] += (
                    1.0 / (rrf_k + rank)
                )

                distance = row.get("_distance")

                if distance is None:
                    distance = row.get("distance")

                if distance is not None:
                    item[f"score_{scale}"] = (
                        float(distance)
                    )

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        output.append([
            {
                "event_id": event_id,
                **payload,
            }
            for event_id, payload in ranked[:top_n]
        ])

    return output


def _metadata_for_event(
    lookup,
    event_id,
):
    return lookup.get_event_metadata(
        int(event_id)
    )


def _window_metadata(
    metadata,
    scale,
):
    window_id = metadata.get( f"{scale}_window_id" )
    token_pos = metadata.get( f"{scale}_window_token_pos" )

    if window_id is not None:
        window_id = int(window_id)

    if token_pos is not None:
        token_pos = int(token_pos)

        if token_pos == _NO_WPOS:
            token_pos = None

    return window_id, token_pos


def _build_batch_events(
    *,
    seed_event_ids,
    neighbours,
    lookup,
    false_positives,
):
    output = []

    for seed_event_id, seed_neighbours in zip(
        seed_event_ids,
        neighbours,
    ):
        seed_event_id = int(seed_event_id)

        seed_metadata = _metadata_for_event(
            lookup,
            seed_event_id,
        )

        neighbours_out = []

        for item in seed_neighbours:
            neighbour_id = int(item["event_id"])

            if neighbour_id == seed_event_id:
                continue

            metadata = _metadata_for_event(
                lookup,
                neighbour_id,
            )

            token = str(metadata["token"])

            if token.lower() in false_positives:
                continue

            (
                local_window_id,
                local_window_token_pos,
            ) = _window_metadata( metadata, "local", )

            (
                medium_window_id,
                medium_window_token_pos,
            ) = _window_metadata( metadata, "medium", )

            (
                broad_window_id,
                broad_window_token_pos,
            ) = _window_metadata( metadata, "broad", )

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "token": token,
                    "doc_id": str( metadata["doc_id"] ),
                    "pub_year": int( metadata["pub_year"] ),
                    "token_idx": int( metadata["token_idx"] ),
                    "local_window_id": ( local_window_id ),
                    "local_window_token_pos": ( local_window_token_pos ),
                    "medium_window_id": ( medium_window_id ),
                    "medium_window_token_pos": ( medium_window_token_pos ),
                    "broad_window_id": ( broad_window_id ),
                    "broad_window_token_pos": ( broad_window_token_pos ),
                    "score": item["score"],
                    "score_local": ( item["score_local"] ),
                    "score_medium": ( item["score_medium"] ),
                    "score_broad": ( item["score_broad"] ),
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        (
            local_window_id,
            local_window_token_pos,
        ) = _window_metadata( seed_metadata, "local", )

        (
            medium_window_id,
            medium_window_token_pos,
        ) = _window_metadata( seed_metadata, "medium", )

        (
            broad_window_id,
            broad_window_token_pos,
        ) = _window_metadata( seed_metadata, "broad", )

        output.append(
            {
                "event_id": seed_event_id,
                "token": str( seed_metadata["token"] ),
                "doc_id": str( seed_metadata["doc_id"] ),
                "pub_year": int( seed_metadata["pub_year"] ),
                "token_idx": int( seed_metadata["token_idx"] ),
                "local_window_id": ( local_window_id ),
                "local_window_token_pos": ( local_window_token_pos ),
                "medium_window_id": ( medium_window_id ),
                "medium_window_token_pos": ( medium_window_token_pos ),
                 "broad_window_id": ( broad_window_id ),
                "broad_window_token_pos": ( broad_window_token_pos ),
                "neighbours": neighbours_out,
            }
        )

    return output


def iter_concept_batches(
    *,
    lookup,
    indexes,
    seed_event_ids,
    scales,
    year_start,
    year_end,
    top_n,
    rrf_k,
    oversample,
    false_positives,
    batch_size,
):
    """
    Yield bounded Tier 2 batches.

    Each scale searches with the corresponding Tier 1 embedding rather than
    with an averaged ensemble vector.
    """
    if not seed_event_ids:
        return

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    # Fetch every scale's embeddings for the whole seed set up front, one
    # batched query per scale, instead of once per BATCH_SIZE-sized chunk
    # (BATCH_SIZE defaults to 5). ParquetObservationLookup answers this
    # with a single `event_id IN (...)` query per scale, so this turns
    # len(seed_event_ids) / batch_size tiny fetches into 3 large ones.
    embeddings_by_scale = {
        scale: lookup.get_scale_embeddings(
            seed_event_ids,
            scale,
        )
        for scale in scales
    }

    # One executor shared across every batch/scale search below, instead
    # of a fresh ThreadPoolExecutor per scale per batch.
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(seed_event_ids), batch_size):
            seed_batch = seed_event_ids[start:start + batch_size]
            end = start + len(seed_batch)

            queries_by_scale = {
                scale: embeddings_by_scale[scale][start:end]
                for scale in scales
            }

            neighbours = multiscale_search(
                indexes=indexes,
                queries_by_scale=queries_by_scale,
                year_start=year_start,
                year_end=year_end,
                scales=scales,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
                executor=executor,
            )

            events = _build_batch_events(
                seed_event_ids=seed_batch,
                neighbours=neighbours,
                lookup=lookup,
                false_positives=false_positives,
            )

            yield {
                "type": "batch",
                "events": events,
            }
