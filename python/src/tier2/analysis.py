"""
tier2/analysis.py
"""

from __future__ import annotations

import numpy as np

from lib.corpus_logging import logger
from tier1.observation_store_api import SCALES
from retrieval.models import INVALID_EVENT_ID

K = 60
RRF_K = 60
OVERSAMPLE = 5
BATCH_SIZE = 128

_NO_WPOS = -1


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

    event_ids = [
        int(event_id)
        for event_id in event_ids
    ]

    logger.info(
        "[tier2] %s: %d seed events",
        concept_name,
        len(event_ids),
    )

    by_year = {}

    for event_id in event_ids:
        metadata = lookup.get_event_metadata(
            event_id
        )
        year = int(metadata["pub_year"])
        by_year.setdefault(
            year,
            [],
        ).append(event_id)

    return {
        "forms": forms,
        "false_positives": false_positives,
        "event_ids": event_ids,
        "event_ids_set": set(event_ids),
        "by_year": by_year,
    }


def multiscale_search(
    *,
    indexes,
    queries_by_scale,
    top_n=K,
    rrf_k=RRF_K,
    oversample=OVERSAMPLE,
    scales=SCALES,
):
    """
    Search each scale using its corresponding Tier 1 embedding.

    The supplied indexes define the temporal search scope. All scales
    therefore search the same candidate population for a query.

    RRF merges scale rankings by stable event_id.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    scales = tuple(scales)

    if not scales:
        raise ValueError("at least one scale is required")

    search_k = top_n * oversample
    results_by_scale = {}

    for scale in scales:
        index = indexes.get(scale)

        if index is None:
            raise RuntimeError(
                f"Missing observation index for scale={scale}"
            )

        queries = np.asarray(
            queries_by_scale[scale],
            dtype=np.float32,
        )

        if queries.ndim != 2:
            raise ValueError(
                f"queries for scale={scale} must be two-dimensional"
            )

        results_by_scale[scale] = index.batch_search(
            queries,
            k=search_k,
            oversample=oversample,
        )

    first_scale = scales[0]

    query_count = len(
        queries_by_scale[first_scale]
    )

    for scale in scales[1:]:
        if len(queries_by_scale[scale]) != query_count:
            raise ValueError(
                "all scales must contain the same number of query vectors"
            )

    output = []

    for query_idx in range(query_count):
        fused = {}

        for scale in scales:
            result = results_by_scale[scale]

            event_ids = result.event_ids[query_idx]
            distances = result.distances[query_idx]

            for rank, (
                event_id,
                distance,
            ) in enumerate(
                zip(
                    event_ids,
                    distances,
                ),
                start=1,
            ):
                event_id = int(event_id)

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
                    1.0
                    / (rrf_k + rank)
                )

                item[
                    f"score_{scale}"
                ] = float(distance)

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
    return lookup.get_event_metadata(
        int(event_id)
    )


def _window_metadata(
    metadata,
    scale,
):
    window_id = metadata.get(
        f"{scale}_window_id"
    )
    token_pos = metadata.get(
        f"{scale}_window_token_pos"
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
            neighbour_id = int(
                item["event_id"]
            )

            if neighbour_id == seed_event_id:
                continue

            metadata = _metadata_for_event(
                lookup,
                neighbour_id,
            )

            token = str(
                metadata["token"]
            )

            if token.lower() in false_positives:
                continue

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
                    "doc_id": str(
                        metadata["doc_id"]
                    ),
                    "pub_year": int(
                        metadata["pub_year"]
                    ),
                    "token_idx": int(
                        metadata["token_idx"]
                    ),
                    "local_window_id": (
                        local_window_id
                    ),
                    "local_window_token_pos": (
                        local_window_token_pos
                    ),
                    "medium_window_id": (
                        medium_window_id
                    ),
                    "medium_window_token_pos": (
                        medium_window_token_pos
                    ),
                    "broad_window_id": (
                        broad_window_id
                    ),
                    "broad_window_token_pos": (
                        broad_window_token_pos
                    ),
                    "score": item["score"],
                    "score_local": (
                        item["score_local"]
                    ),
                    "score_medium": (
                        item["score_medium"]
                    ),
                    "score_broad": (
                        item["score_broad"]
                    ),
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
                "token": str(
                    seed_metadata["token"]
                ),
                "doc_id": str(
                    seed_metadata["doc_id"]
                ),
                "pub_year": int(
                    seed_metadata["pub_year"]
                ),
                "token_idx": int(
                    seed_metadata["token_idx"]
                ),
                "local_window_id": (
                    local_window_id
                ),
                "local_window_token_pos": (
                    local_window_token_pos
                ),
                "medium_window_id": (
                    medium_window_id
                ),
                "medium_window_token_pos": (
                    medium_window_token_pos
                ),
                "broad_window_id": (
                    broad_window_id
                ),
                "broad_window_token_pos": (
                    broad_window_token_pos
                ),
                "neighbours": neighbours_out,
            }
        )

    return output


def iter_concept_batches(
    *,
    lookup,
    indexes_by_year,
    seed_event_ids,
    scales,
    top_n,
    rrf_k,
    oversample,
    false_positives,
    batch_size,
):
    """
    Yield bounded Tier 2 batches.

    Each seed is searched only against observations from the seed's
    publication year. Multiscale retrieval and RRF therefore operate
    within that temporal population.

    Failure mode:
        A seed year without a corresponding index cannot produce neighbours.
        This should only occur if the caller constructs indexes inconsistently
        with the resolved candidate-year set.
    """
    if not seed_event_ids:
        return

    false_positives = {
        str(value).lower()
        for value in (false_positives or [])
    }

    embeddings_by_scale = {
        scale: lookup.get_scale_embeddings(
            seed_event_ids,
            scale,
        )
        for scale in scales
    }

    seed_years = []

    for event_id in seed_event_ids:
        metadata = lookup.get_event_metadata(
            int(event_id)
        )

        seed_years.append(
            int(metadata["pub_year"])
        )

    for start in range(
        0,
        len(seed_event_ids),
        batch_size,
    ):
        seed_batch = seed_event_ids[
            start:start + batch_size
        ]

        batch_years = seed_years[
            start:start + len(seed_batch)
        ]

        queries_by_scale = {
            scale: embeddings_by_scale[scale][
                start:start + len(seed_batch)
            ]
            for scale in scales
        }

        batch_events = [
            None
            for _ in seed_batch
        ]

        year_groups = {}

        for local_index, year in enumerate(batch_years):
            year_groups.setdefault(
                year,
                [],
            ).append(local_index)

        for year, local_indices in year_groups.items():
            indexes = indexes_by_year.get(year)

            if indexes is None:
                raise RuntimeError(
                    f"No temporal indexes available for publication year "
                    f"{year}"
                )

            year_queries_by_scale = {
                scale: queries_by_scale[scale][
                    local_indices
                ]
                for scale in scales
            }

            neighbours = multiscale_search(
                indexes=indexes,
                queries_by_scale=year_queries_by_scale,
                scales=scales,
                top_n=top_n,
                rrf_k=rrf_k,
                oversample=oversample,
            )

            year_seed_ids = [
                seed_batch[index]
                for index in local_indices
            ]

            events = _build_batch_events(
                seed_event_ids=year_seed_ids,
                neighbours=neighbours,
                lookup=lookup,
                false_positives=false_positives,
            )

            for local_index, event in zip(
                local_indices,
                events,
            ):
                batch_events[local_index] = event

        yield {
            "type": "batch",
            "events": batch_events,
        }

