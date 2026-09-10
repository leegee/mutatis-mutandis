from __future__ import annotations

import numpy as np

from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.models import INVALID_EVENT_ID


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Kept for backward compatibility / single-list callers."""
    scores: dict[int, float] = {}

    for ranked in ranked_lists:
        for rank, event_id in enumerate(ranked, start=1):
            if event_id == INVALID_EVENT_ID:
                continue
            scores[event_id] = scores.get(event_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return fused[:top_n] if top_n is not None else fused


def _lookup_distance(
    all_ids: np.ndarray,
    all_dists: np.ndarray,
    scale_idx: int,
    query_idx: int,
    event_id: int,
) -> float | None:
    """Return the distance for (scale, query, event_id) or None."""
    mask = all_ids[scale_idx, query_idx] == event_id
    if np.any(mask):
        return float(all_dists[scale_idx, query_idx][mask][0])
    return None


def multiscale_search(
    neighbour_indexes: dict[str, LanceObservationIndex],
    queries_by_scale: dict[str, np.ndarray],
    scales: tuple[str, ...],
    top_n: int,
    *,
    rrf_k: int = 60,
    oversample: int | float = 1,
    exclude_event_ids: tuple[int, ...] | None = None,
) -> list[list[dict]]:
    """
    Search an explicit neighbour population and fuse its rankings with RRF.

    This version keeps intermediate results in NumPy arrays for as long as
    possible and only materialises Python dicts for the final per-seed
    result lists that the rest of the Tier-2 pipeline expects.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if oversample <= 0:
        raise ValueError("oversample must be positive")
    if not scales:
        raise ValueError("at least one scale is required")

    # ------------------------------------------------------------------
    # 1. Run the per-scale batch searches (still the dominant cost)
    # ------------------------------------------------------------------
    per_scale: dict[str, object] = {}
    query_count: int | None = None

    for scale in scales:
        index = neighbour_indexes.get(scale)
        if index is None:
            raise KeyError(f"Missing neighbour index for scale={scale}")

        queries = queries_by_scale.get(scale)
        if queries is None:
            raise KeyError(f"Missing queries for scale={scale}")

        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError(
                f"queries for scale={scale} must be two-dimensional"
            )

        if query_count is None:
            query_count = queries.shape[0]
        elif queries.shape[0] != query_count:
            raise ValueError(
                "all scale query arrays must contain the same number of queries"
            )

        per_scale[scale] = index.batch_search(
            queries,
            k=top_n,
            oversample=oversample,
        )

    if query_count is None or query_count == 0:
        return []

    N = query_count
    S = len(scales)
    K = top_n

    # ------------------------------------------------------------------
    # 2. Stack into contiguous NumPy arrays: shape (S, N, K)
    # ------------------------------------------------------------------
    all_ids = np.stack(
        [per_scale[s].event_ids for s in scales],
        axis=0,
    )  # uint64
    all_dists = np.stack(
        [per_scale[s].distances for s in scales],
        axis=0,
    )  # float32

    # ------------------------------------------------------------------
    # 3. Build validity mask (sentinel + optional exclusion)
    # ------------------------------------------------------------------
    is_sentinel = all_ids == INVALID_EVENT_ID

    if exclude_event_ids is not None:
        if len(exclude_event_ids) != N:
            raise ValueError(
                "exclude_event_ids must contain exactly one event ID per query"
            )
        excl = np.asarray(exclude_event_ids, dtype=np.uint64)  # (N,)
        is_excluded = all_ids == excl[None, :, None]           # (S, N, K)
    else:
        is_excluded = np.zeros_like(all_ids, dtype=bool)

    valid = ~(is_sentinel | is_excluded)  # (S, N, K)

    # ------------------------------------------------------------------
    # 4. Per-query RRF (still a Python loop over N, but everything
    #    inside stays in NumPy / small dicts)
    # ------------------------------------------------------------------
    fused: list[list[dict]] = []

    # Pre-compute scale index for the three well-known names
    scale_to_idx = {name: i for i, name in enumerate(scales)}

    for q in range(N):
        # Gather every valid (event_id, scale_rank) pair for this query
        # We build a small dict: event_id → list of ranks (one per scale)
        ranks_by_id: dict[int, list[int]] = {}

        for s in range(S):
            mask = valid[s, q]                     # (K,)
            if not np.any(mask):
                continue

            ids_s = all_ids[s, q, mask]            # 1-D
            # ranks start at 1 and follow the order already present
            # (batch_search returns results sorted by distance)
            ranks_s = np.arange(1, len(ids_s) + 1, dtype=np.int32)

            for eid, rank in zip(ids_s.tolist(), ranks_s.tolist()):
                ranks_by_id.setdefault(int(eid), []).append(rank)

        if not ranks_by_id:
            fused.append([])
            continue

        # Compute RRF scores
        scored = []
        for eid, rank_list in ranks_by_id.items():
            score = sum(1.0 / (rrf_k + r) for r in rank_list)
            scored.append((eid, score))

        # Sort descending by score and keep top_n
        scored.sort(key=lambda t: t[1], reverse=True)
        top = scored[:top_n]

        # Materialise the final dicts (only place we leave pure NumPy)
        q_result = []
        for rank, (eid, rrf_score) in enumerate(top, start=1):
            q_result.append({
                "event_id": eid,
                "rank": rank,
                "rrf_score": float(rrf_score),
                "score": float(rrf_score),
                "score_local": _lookup_distance(
                    all_ids, all_dists,
                    scale_to_idx.get("local", -1), q, eid
                ) if "local" in scale_to_idx else None,
                "score_medium": _lookup_distance(
                    all_ids, all_dists,
                    scale_to_idx.get("medium", -1), q, eid
                ) if "medium" in scale_to_idx else None,
                "score_broad": _lookup_distance(
                    all_ids, all_dists,
                    scale_to_idx.get("broad", -1), q, eid
                ) if "broad" in scale_to_idx else None,
            })

        fused.append(q_result)

    return fused
