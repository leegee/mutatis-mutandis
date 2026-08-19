from __future__ import annotations

from collections import defaultdict

import numpy as np

from lib.corpus_logging import logger

from .models import Float32Array


DEFAULT_RRF_K = 60
DEFAULT_OVERSAMPLE = 5
DEFAULT_SCALE_WEIGHTS = {
    "local": 0.25,
    "medium": 0.50,
    "broad": 0.25,
}


def multiscale_diskann_search(
    *,
    indexes,
    lookup,
    positions: np.ndarray,
    top_n: int,
    pub_year: int,
    rrf_k: int = DEFAULT_RRF_K,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> list[list[dict]]:
    """
    Retrieve and fuse neighbours for seed observations in one year.

    Each scale searches its own DiskANN graph. Candidates are fused by
    reciprocal rank fusion, while the existing score_local,
    score_medium, score_broad and score field names are retained for
    downstream compatibility.

    `positions` contains dense ObservationLookup positions, not DiskANN
    identifiers. event_id remains the stable identity throughout retrieval.
    """
    positions = np.asarray(positions, dtype=np.int64)

    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")

    if len(positions) == 0:
        return []

    if top_n <= 0:
        raise ValueError("top_n must be positive")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    if oversample <= 0:
        raise ValueError("oversample must be positive")

    year_indexes = indexes[pub_year]

    event_ids = np.asarray(
        lookup.event_id[positions],
        dtype=np.uint64,
    )

    search_k = top_n * oversample

    per_scale = {}

    for scale in ("local", "medium", "broad"):
        index = year_indexes[scale]

        queries = lookup.get_scale_embeddings(
            event_ids,
            scale,
        )

        results = index.batch_search(
            queries,
            k=search_k,
        )

        per_scale[scale] = results

        logger.debug(
            "[tier2] year=%s scale=%s queries=%d k=%d",
            pub_year,
            scale,
            len(event_ids),
            search_k,
        )

    return _fuse_results(
        event_ids=event_ids,
        per_scale=per_scale,
        top_n=top_n,
        rrf_k=rrf_k,
    )


def _fuse_results(
    *,
    event_ids: np.ndarray,
    per_scale,
    top_n: int,
    rrf_k: int,
) -> list[list[dict]]:
    """
    Convert three scale-specific ANN result matrices into per-query RRF
    rankings.

    DiskANN returns distances, where smaller means closer. RRF itself uses
    rank rather than distance, so distance is retained only as the
    scale-specific diagnostic score.
    """
    fused = []

    for query_idx, seed_event_id in enumerate(event_ids):
        candidates = {}

        for scale in ("local", "medium", "broad"):
            result = per_scale[scale].row(query_idx)

            candidate_ids = result.event_ids
            distances = result.distances

            for rank, (event_id, distance) in enumerate(
                zip(candidate_ids, distances),
                start=1,
            ):
                event_id = int(event_id)

                if event_id == int(seed_event_id):
                    continue

                item = candidates.setdefault(
                    event_id,
                    {
                        "event_id": event_id,
                        "score_local": None,
                        "score_medium": None,
                        "score_broad": None,
                        "_rrf": 0.0,
                    },
                )

                item[f"score_{scale}"] = float(distance)

                item["_rrf"] += 1.0 / (
                    rrf_k + rank
                )

        ranked = sorted(
            candidates.values(),
            key=lambda item: item["_rrf"],
            reverse=True,
        )[:top_n]

        output = []

        for item in ranked:
            output.append(
                {
                    "event_id": item["event_id"],
                    "score": item["_rrf"],
                    "score_local": item["score_local"],
                    "score_medium": item["score_medium"],
                    "score_broad": item["score_broad"],
                }
            )

        fused.append(output)

    return fused
