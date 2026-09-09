from __future__ import annotations

import numpy as np

from retrieval.lance_observation_index import LanceObservationIndex


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}

    for ranked in ranked_lists:
        for rank, event_id in enumerate(
            ranked,
            start=1,
        ):
            if event_id == -1:
                continue

            scores[event_id] = (
                scores.get(event_id, 0.0)
                + 1.0 / (k + rank)
            )

    fused = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return (
        fused[:top_n]
        if top_n is not None
        else fused
    )


def multiscale_search(
    neighbour_indexes: dict[str, LanceObservationIndex],
    queries_by_scale: dict[str, np.ndarray],
    scales: tuple[str, ...],
    top_n: int,
    *,
    rrf_k: int = 60,
    oversample: int = 5,
    exclude_event_ids: tuple[int, ...] | None = None,
) -> list[list[dict]]:
    """
    Search an explicit neighbour population and fuse its rankings with RRF.

    Query vectors belong to the seed population and are supplied separately.
    neighbour_indexes define the observations eligible to answer those
    queries.

    Each query may exclude its own seed event. Exclusion occurs before RRF
    so that a seed cannot consume a retrieval slot or contribute to its
    fused ranking.

    Failure modes:
        A missing scale index or query array is an explicit configuration
        error.

        All scales must contain the same number of query vectors. Otherwise
        results could be associated with the wrong seed.

        If exclude_event_ids is supplied, it must contain exactly one event
        ID per query.

        Oversampling is applied by the observation index before population
        exclusions so that excluded candidates do not unnecessarily reduce
        the final top_n.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    if not scales:
        raise ValueError("at least one scale is required")

    per_scale = {}
    query_count = None

    for scale in scales:
        index = neighbour_indexes.get(scale)

        if index is None:
            raise KeyError(
                f"Missing neighbour index for scale={scale}"
            )

        queries = queries_by_scale.get(scale)

        if queries is None:
            raise KeyError(
                f"Missing queries for scale={scale}"
            )

        queries = np.asarray(
            queries,
            dtype=np.float32,
        )

        if queries.ndim != 2:
            raise ValueError(
                f"queries for scale={scale} must be two-dimensional"
            )

        if query_count is None:
            query_count = queries.shape[0]
        elif queries.shape[0] != query_count:
            raise ValueError(
                "all scale query arrays must contain the same "
                "number of queries"
            )

        per_scale[scale] = index.batch_search(
            queries,
            k=top_n,
            oversample=oversample,
        )

    if query_count is None:
        return []

    if exclude_event_ids is not None:
        if len(exclude_event_ids) != query_count:
            raise ValueError(
                "exclude_event_ids must contain exactly one "
                "event ID per query"
            )

        excluded_by_query = tuple(
            int(event_id)
            for event_id in exclude_event_ids
        )
    else:
        excluded_by_query = None

    fused = []

    for query_index in range(query_count):
        excluded_event_id = (
            excluded_by_query[query_index]
            if excluded_by_query is not None
            else None
        )

        scale_scores = {}

        for scale in scales:
            result = per_scale[scale]

            scale_scores[scale] = {
                int(event_id): float(score)
                for event_id, score in zip(
                    result.event_ids[query_index],
                    result.distances[query_index],
                )
                if (
                    int(event_id) != -1
                    and int(event_id) != excluded_event_id
                )
            }

        ranked_lists = [
            list(scale_scores[scale].keys())
            for scale in scales
        ]

        fused_ids = reciprocal_rank_fusion(
            ranked_lists,
            k=rrf_k,
            top_n=top_n,
        )

        fused.append(
            [
                {
                    "event_id": event_id,
                    "rank": rank,
                    "rrf_score": rrf_score,
                    "score": rrf_score,
                    "score_local": (
                        scale_scores.get(
                            "local",
                            {},
                        ).get(event_id)
                    ),
                    "score_medium": (
                        scale_scores.get(
                            "medium",
                            {},
                        ).get(event_id)
                    ),
                    "score_broad": (
                        scale_scores.get(
                            "broad",
                            {},
                        ).get(event_id)
                    ),
                }
                for rank, (event_id, rrf_score)
                in enumerate(
                    fused_ids,
                    start=1,
                )
            ]
        )

    return fused
