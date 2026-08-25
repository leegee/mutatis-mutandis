"""
tier2/analysis.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES

K = 200
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 2000

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
    """
    Resolve lexical seed events and group their stable event IDs by year.

    Tier 2 uses event_id as its identity; physical vector positions are not
    part of the retrieval contract.

    Failure mode:
        Metadata lookup is intentionally performed only for seed events.
        The potentially very large observation corpus is never scanned into
        Python merely to construct the year grouping.
    """
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

    by_year: dict[int, list[int]] = {}

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
    year_start: int | None,
    year_end: int | None,
    model: str | None = None,
) -> str | None:
    clauses = []

    if year_start is not None:
        clauses.append(f"year >= {int(year_start)}")

    if year_end is not None:
        clauses.append(f"year <= {int(year_end)}")

    if model is not None:
        escaped = model.replace("'", "''")
        clauses.append(f"embedding_model = '{escaped}'")

    return " AND ".join(clauses) if clauses else None


def _search_lance_table(
    *,
    table,
    queries,
    year_start,
    year_end,
    k,
    nprobes,
):
    """
    Search a Lance scale table for a batch of query vectors.

    Lance performs the temporal restriction at query time. The physical
    vector index therefore remains corpus-wide.

    Failure mode:
        Lance's Python API is query-oriented rather than DiskANN's batched
        native API, so results are collected one query at a time here.
        Keep this implementation simple until retrieval profiling identifies
        this as a bottleneck.
    """
    queries = np.asarray(
        queries,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError("queries must be two-dimensional")

    results = []

    where = _lance_where(
        year_start=year_start,
        year_end=year_end,
    )

    for query in queries:
        search = (
            table
            .search(query)
            .nprobes(nprobes)
        )

        if where:
            search = search.where(where)

        rows = search.limit(k).to_list()

        results.append(rows)

    return results


def multiscale_search(
    *,
    indexes,
    queries,
    year_start: int | None,
    year_end: int | None,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    scales=SCALES,
    nprobes=20,
):
    """
    Search the selected Lance scale tables and fuse their rankings.

    The year range applies to every scale search. RRF is performed by stable
    event_id rather than by Lance's physical row identity.

    Failure mode:
        A neighbour may occur in multiple scale-specific result sets.
        RRF therefore deliberately merges those occurrences by event_id.
    """
    queries = np.asarray(
        queries,
        dtype=np.float32,
    )

    if queries.ndim != 2:
        raise ValueError("queries must be two-dimensional")

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    search_k = top_n * oversample

    results_by_scale = {}

    for scale in scales:
        table = indexes.get(scale)

        if table is None:
            raise RuntimeError(
                f"Missing Lance table for scale={scale}"
            )

        results_by_scale[scale] = _search_lance_table(
            table=table,
            queries=queries,
            year_start=year_start,
            year_end=year_end,
            k=search_k,
            nprobes=nprobes,
        )

    output = []

    for query_idx in range(len(queries)):
        fused = {}

        for scale in scales:
            result = results_by_scale[scale][query_idx]

            for rank, row in enumerate(result, start=1):
                if "event_id" not in row:
                    raise RuntimeError(
                        f"Lance result for scale={scale} has no event_id"
                    )

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

                item["score"] += 1.0 / (rrf_k + rank)

                distance = row.get("_distance")

                if distance is None:
                    distance = row.get("distance")

                if distance is not None:
                    item[f"score_{scale}"] = float(distance)

        ranked = sorted(
            fused.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )

        output.append(
            [
                {
                    "event_id": event_id,
                    **payload,
                }
                for event_id, payload in ranked[:top_n]
            ]
        )

    return output


def _metadata_for_event(
    lookup,
    event_id,
):
    """Return public Tier 1 metadata for one stable observation ID."""
    return lookup.get_event_metadata(int(event_id))


def _window_metadata(
    metadata,
    scale,
):
    window_id = metadata.get(
        f"{scale}_window_id",
    )

    token_pos = metadata.get(
        f"{scale}_window_token_pos",
    )

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
    """
    Materialise one bounded batch in the existing Tier 2 event schema.

    Only metadata for the seeds and their selected neighbours is requested.
    Embeddings are not reconstructed here.
    """
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

            doc_id = str(metadata["doc_id"])

            (
                local_window_id,
                local_window_token_pos,
            ) = _window_metadata(
                metadata,
                "local",
            )

            (
                medium_window_id,
                medium_window_token_pos,
            ) = _window_metadata(
                metadata,
                "medium",
            )

            (
                broad_window_id,
                broad_window_token_pos,
            ) = _window_metadata(
                metadata,
                "broad",
            )

            neighbours_out.append(
                {
                    "event_id": neighbour_id,
                    "token": token,
                    "doc_id": doc_id,
                    "pub_year": int(metadata["pub_year"]),
                    "token_idx": int(metadata["token_idx"]),
                    "local_window_id": local_window_id,
                    "local_window_token_pos": local_window_token_pos,
                    "medium_window_id": medium_window_id,
                    "medium_window_token_pos": medium_window_token_pos,
                    "broad_window_id": broad_window_id,
                    "broad_window_token_pos": broad_window_token_pos,
                    "score": item["score"],
                    "score_local": item["score_local"],
                    "score_medium": item["score_medium"],
                    "score_broad": item["score_broad"],
                    "depth": 1,
                    "via_event_id": None,
                }
            )

        (
            local_window_id,
            local_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "local",
        )

        (
            medium_window_id,
            medium_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "medium",
        )

        (
            broad_window_id,
            broad_window_token_pos,
        ) = _window_metadata(
            seed_metadata,
            "broad",
        )

        output.append(
            {
                "event_id": seed_event_id,
                "token": str(seed_metadata["token"]),
                "doc_id": str(seed_metadata["doc_id"]),
                "pub_year": int(seed_metadata["pub_year"]),
                "token_idx": int(seed_metadata["token_idx"]),
                "local_window_id": local_window_id,
                "local_window_token_pos": local_window_token_pos,
                "medium_window_id": medium_window_id,
                "medium_window_token_pos": medium_window_token_pos,
                "broad_window_id": broad_window_id,
                "broad_window_token_pos": broad_window_token_pos,
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
    Yield bounded Tier 2 batches for seeds searched across one SearchSpace.

    Seeds remain grouped by their publication year for bookkeeping, but Lance
    searches each seed against the complete requested temporal window.
    """
    if not seed_event_ids:
        return

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    for seed_batch in _chunks(
        seed_event_ids,
        batch_size,
    ):
        queries = lookup.get_embeddings(
            seed_batch,
            scales=scales,
        )

        neighbours = multiscale_search(
            indexes=indexes,
            queries=queries,
            year_start=year_start,
            year_end=year_end,
            scales=scales,
            top_n=top_n,
            rrf_k=rrf_k,
            oversample=oversample,
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
