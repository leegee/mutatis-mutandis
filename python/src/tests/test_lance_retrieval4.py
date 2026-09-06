#!/usr/bin/env python
"""
Benchmark bucket-exact Parquet retrieval against LanceDB retrieval.

The benchmark mirrors the production Tier 2 retrieval pattern:

    one chronological bucket
        -> local / medium / broad search
        -> oversampled candidate lists
        -> reciprocal-rank fusion
        -> final top-N neighbours

Exact Parquet search is the ground truth. Every observation in the selected
bucket is scored, so the exact and ANN systems operate over identical
chronological populations.

The benchmark therefore answers the operational question:

    "If Tier 2 uses Lance instead of bucket-exact Parquet, how often does the
     final RRF neighbourhood change?"

Usage
-----
    python src/tests/test_lance_retrieval4.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from lib.corpus_config import LANCE_INDEXES_DIR

from lib.corpus_logging import logger
from tier1.observation_store_api import (
    SCALES,
    default_store_path,
    open_observation_lookup,
)
from retrieval.exact_knn_search import (
    Shard,
    _filter_shards,
    _query_vectors_for_positions,
    _reciprocal_rank_fusion,
    discover_shards,
    exact_knn,
)
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from retrieval.models import SearchSpace

TOP_N = 60
OVERSAMPLE = 5
RRF_K = 60

# Keep this explicit for the benchmark so a single run can isolate a known
# chronological population. Production retrieval should derive these bounds
# from the seed event and chronology configuration.
BUCKET_START = 1540
BUCKET_END = 1549

SEED_FORMS = (
    "white",
    "pale",
    "bright",
    "blond",
    "hoary",
    "albino",
)

SEEDS_PER_FORM = 1

NPROBES = 150


def recall_at_k(
    exact_ids: list[int],
    ann_ids: list[int],
    k: int,
) -> float:
    """
    Recall of the ANN top-k against the exact top-k.

    This remains a diagnostic. Final fused overlap is the primary benchmark
    because that is what Tier 2 actually consumes.
    """
    exact = set(exact_ids[:k])
    ann = set(ann_ids[:k])

    if not exact:
        return 1.0

    return len(exact & ann) / len(exact)


def overlap_at_k(
    exact_ids: list[int],
    ann_ids: list[int],
    k: int,
) -> float:
    """Jaccard overlap diagnostic for the final neighbourhood."""
    exact = set(exact_ids[:k])
    ann = set(ann_ids[:k])

    if not exact and not ann:
        return 1.0

    union = exact | ann

    if not union:
        return 1.0

    return len(exact & ann) / len(union)


def _ranked_scale_lists(
    scores: np.ndarray,
    ids: np.ndarray,
) -> list[list[int]]:
    """Convert one scale's fixed-width result matrix into ranked IDs."""
    ranked: list[list[int]] = []

    for row_ids, row_scores in zip(ids, scores):
        ranked.append(
            [
                int(event_id)
                for event_id, score in zip(row_ids, row_scores)
                if int(event_id) >= 0 and np.isfinite(score)
            ]
        )

    return ranked


def fuse_results(
    per_scale_results: dict[str, tuple[np.ndarray, np.ndarray]],
    top_n: int,
    rrf_k: int,
) -> list[list[dict]]:
    """
    Apply the same RRF operation used by Tier 2.

    Individual scale scores are retained for diagnostics. Ranking is
    determined solely by the ranked event-id lists.
    """
    n_queries = next(
        iter(per_scale_results.values())
    )[0].shape[0]

    fused: list[list[dict]] = []

    ranked_by_scale = {
        scale: _ranked_scale_lists(scores, ids)
        for scale, (scores, ids) in per_scale_results.items()
    }

    for query_index in range(n_queries):
        ranked_lists = [
            ranked_by_scale[scale][query_index]
            for scale in SCALES
        ]

        fused_ids = _reciprocal_rank_fusion(
            ranked_lists,
            k=rrf_k,
            top_n=top_n,
        )

        score_maps = {
            scale: {
                int(event_id): float(score)
                for event_id, score in zip(
                    per_scale_results[scale][1][query_index],
                    per_scale_results[scale][0][query_index],
                )
                if int(event_id) >= 0 and np.isfinite(score)
            }
            for scale in SCALES
        }

        fused.append(
            [
                {
                    "event_id": event_id,
                    "rrf_score": rrf_score,
                    "score_local": score_maps["local"].get(event_id),
                    "score_medium": score_maps["medium"].get(event_id),
                    "score_broad": score_maps["broad"].get(event_id),
                }
                for event_id, rrf_score in fused_ids
            ]
        )

    return fused


def run_bucket_search(
    *,
    queries_by_scale: dict[str, np.ndarray],
    query_event_ids: np.ndarray,
    shards: list[Shard],
    lance_indexes,
    top_n: int,
    oversample: int,
    rrf_k: int,
    nprobes: int,
    exact: bool,
):
    """
    Run the complete three-scale retrieval pipeline inside one bucket.

    Exact and Lance receive the same per-scale query vectors and the same
    chronological candidate population.
    """
    del nprobes

    search_k = top_n * oversample

    per_scale_results: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ] = {}

    for scale in SCALES:
        print(
            f"    {scale}: {'exact' if exact else 'Lance'} "
            f"top-{search_k}..."
        )

        queries = queries_by_scale[scale]

        if exact:
            scores, ids = exact_knn(
                queries,
                shards,
                k=search_k,
                scale=scale,
                workers=1,
                pool="thread",
                exclude_self=True,
                query_event_ids=query_event_ids.tolist(),
            )

        else:
            index = lance_indexes[scale]

            rows = []

            for query, query_event_id in zip(
                queries,
                query_event_ids,
            ):
                result = index.search(
                    query,
                    k=search_k + 1,
                )

                filtered_ids = []
                filtered_scores = []

                for event_id, score in zip(
                    result.event_ids,
                    result.distances,
                ):
                    event_id = int(event_id)

                    if event_id == int(query_event_id):
                        continue

                    filtered_ids.append(event_id)
                    filtered_scores.append(float(score))

                    if len(filtered_ids) >= search_k:
                        break

                row_ids = np.full(
                    search_k,
                    -1,
                    dtype=np.uint64,
                )

                row_scores = np.full(
                    search_k,
                    -np.inf,
                    dtype=np.float32,
                )

                width = min(
                    search_k,
                    len(filtered_ids),
                )

                if width:
                    row_ids[:width] = np.asarray(
                        filtered_ids[:width],
                        dtype=np.uint64,
                    )

                    row_scores[:width] = np.asarray(
                        filtered_scores[:width],
                        dtype=np.float32,
                    )

                rows.append(
                    (row_scores, row_ids)
                )

            scores = np.stack(
                [row[0] for row in rows],
                axis=0,
            )

            ids = np.stack(
                [row[1] for row in rows],
                axis=0,
            )

        per_scale_results[scale] = (
            scores,
            ids,
        )

    return fuse_results(
        per_scale_results,
        top_n=top_n,
        rrf_k=rrf_k,
    )


def main():
    print("Opening Tier 1 observation lookup...")

    lookup_root = default_store_path()

    print(
        f"Tier 1 root: {lookup_root}"
    )

    lookup = open_observation_lookup(
        lookup_root
    )

    print(
        f"Tier 1 observations: "
        f"{len(lookup):,}"
    )

    available_years = {
        int(year)
        for year in lookup.available_years
    }

    print(
        f"Available years: "
        f"{min(available_years)}-"
        f"{max(available_years)}"
    )

    print("\nOpening Lance indexes...")

    lance_root = LANCE_INDEXES_DIR

    lance_store = LanceObservationIndexStore(
        lance_root,
        available_years=available_years,
        nprobes=NPROBES,
    )

    print(
        "Opening exact-search Parquet shards..."
    )

    parquet_root = default_store_path()

    all_shards = discover_shards(
        parquet_root
    )

    print(
        f"Parquet shards: {len(all_shards)}"
    )

    print("\nChecking Lance indexes...")

    search_space = SearchSpace(
        years=(BUCKET_START, BUCKET_END),
        scale=SCALES,
    )

    lance_indexes = lance_store.get(
        search_space
    )

    for scale in SCALES:
        print(
            f"  {scale:>6}: index opened successfully"
        )

    bucket_shards = [
        shard
        for shard in all_shards
        if shard.year is not None
        and BUCKET_START <= shard.year <= BUCKET_END
    ]

    if not bucket_shards:
        raise RuntimeError(
            f"No Parquet shards found for bucket "
            f"{BUCKET_START}-{BUCKET_END}"
        )

    bucket_rows = sum(
        shard.n_rows
        for shard in bucket_shards
    )

    print(
        f"\nBUCKET"
        f"\n  years: {BUCKET_START}-{BUCKET_END}"
        f"\n  shards: {len(bucket_shards)}"
        f"\n  observations: {bucket_rows:,}"
    )

    print(
        "\nSampling seed occurrences for: "
        + ", ".join(SEED_FORMS)
    )

    print(
        f"Seeds per form: {SEEDS_PER_FORM}"
    )

    positions = []

    for form in SEED_FORMS:
        matches = np.asarray(
            lookup.find_matching_event_ids(
                [form]
            ),
            dtype=np.int64,
        )

        if len(matches) == 0:
            print(
                f"WARNING: no occurrences found "
                f"for '{form}'"
            )
            continue

        bucket_matches = []

        for event_id in matches:
            metadata = lookup.get_event_metadata(
                int(event_id)
            )

            year = int(
                metadata["pub_year"]
            )

            if (
                BUCKET_START
                <= year
                <= BUCKET_END
            ):
                bucket_matches.append(
                    int(event_id)
                )

        if not bucket_matches:
            print(
                f"WARNING: no occurrences found "
                f"for '{form}' in bucket "
                f"{BUCKET_START}-{BUCKET_END}"
            )
            continue

        for event_id in bucket_matches[
            :SEEDS_PER_FORM
        ]:
            positions.append(
                lookup.get_pos(event_id)
            )

    print(
        f"Selected {len(positions)} seed occurrences."
    )

    if not positions:
        raise RuntimeError(
            "No seed occurrences found in benchmark bucket"
        )

    positions = np.asarray(
        positions,
        dtype=np.int64,
    )

    print("\nFetching seed vectors...")

    queries_by_scale = {}
    query_event_ids = None

    for scale in SCALES:
        queries, qids = (
            _query_vectors_for_positions(
                lookup,
                positions,
                scale,
                parquet_root,
                all_shards,
            )
        )

        queries_by_scale[scale] = queries

        if query_event_ids is None:
            query_event_ids = qids
        elif not np.array_equal(
            query_event_ids,
            qids,
        ):
            raise RuntimeError(
                "Seed event IDs differ between scale "
                "query-vector loads"
            )

    assert query_event_ids is not None

    print("Seed vectors loaded.")

    for position, event_id in zip(
        positions,
        query_event_ids,
    ):
        metadata = lookup.get_event_metadata(
            int(event_id)
        )

        print(
            f"  {int(event_id)}: "
            f"'{metadata['token']}' "
            f"{metadata['doc_id']} "
            f"{metadata['pub_year']}"
        )

    print(
        "\nBUCKET-EXACT VS LANCE RRF BENCHMARK"
    )
    print("=" * 78)

    print(
        f"\nBucket: {BUCKET_START}-{BUCKET_END}"
        f"\nExact candidate count per scale: "
        f"{TOP_N * OVERSAMPLE}"
        f"\nLance candidate count per scale: "
        f"{TOP_N * OVERSAMPLE}"
        f"\nFinal RRF top-N: {TOP_N}"
        f"\nRRF_K: {RRF_K}"
        f"\nLance nprobes: {NPROBES}"
    )

    print(
        "\nBoth systems search exactly the same bucket."
    )

    print(
        "Exact = every vector in the bucket is scored."
    )

    print(
        "Each scale receives its own query vector."
    )

    print(
        "Primary metric = overlap of final RRF top-N."
    )

    started = time.perf_counter()

    exact_started = time.perf_counter()

    exact_fused = run_bucket_search(
        queries_by_scale=queries_by_scale,
        query_event_ids=query_event_ids,
        shards=bucket_shards,
        lance_indexes=lance_indexes,
        top_n=TOP_N,
        oversample=OVERSAMPLE,
        rrf_k=RRF_K,
        nprobes=NPROBES,
        exact=True,
    )

    exact_elapsed = (
        time.perf_counter()
        - exact_started
    )

    print(
        f"\nExact bucket search: "
        f"{exact_elapsed:.3f}s"
    )

    lance_started = time.perf_counter()

    lance_fused = run_bucket_search(
        queries_by_scale=queries_by_scale,
        query_event_ids=query_event_ids,
        shards=bucket_shards,
        lance_indexes=lance_indexes,
        top_n=TOP_N,
        oversample=OVERSAMPLE,
        rrf_k=RRF_K,
        nprobes=NPROBES,
        exact=False,
    )

    lance_elapsed = (
        time.perf_counter()
        - lance_started
    )

    print(
        f"\nLance bucket search: "
        f"{lance_elapsed:.3f}s"
    )

    print("\nFINAL RRF COMPARISON")
    print("=" * 78)

    overlaps = []

    for query_index, (
        exact_rows,
        lance_rows,
    ) in enumerate(
        zip(
            exact_fused,
            lance_fused,
        )
    ):
        seed_id = int(
            query_event_ids[query_index]
        )

        exact_ids = [
            int(row["event_id"])
            for row in exact_rows
        ]

        lance_ids = [
            int(row["event_id"])
            for row in lance_rows
        ]

        exact_set = set(exact_ids)
        lance_set = set(lance_ids)

        common = (
            exact_set
            & lance_set
        )

        recall = (
            len(common)
            / len(exact_set)
            if exact_set
            else 1.0
        )

        jaccard = overlap_at_k(
            exact_ids,
            lance_ids,
            TOP_N,
        )

        overlaps.append(
            recall
        )

        print(
            f"\nSeed event_id={seed_id}"
        )

        print(
            f"  exact RRF results: {len(exact_ids)}"
        )

        print(
            f"  Lance RRF results: {len(lance_ids)}"
        )

        print(
            f"  common:             {len(common)}"
        )

        print(
            f"  recall@{TOP_N}:       {recall:.2%}"
        )

        print(
            f"  Jaccard@{TOP_N}:      {jaccard:.2%}"
        )

        if exact_ids != lance_ids:
            print(
                "  ranking identical:   NO"
            )
        else:
            print(
                "  ranking identical:   YES"
            )

        missing = [
            event_id
            for event_id in exact_ids
            if event_id not in lance_set
        ]

        added = [
            event_id
            for event_id in lance_ids
            if event_id not in exact_set
        ]

        if missing:
            print(
                "  missing from Lance:  "
                + ", ".join(
                    str(event_id)
                    for event_id in missing[:10]
                )
            )

        if added:
            print(
                "  added by Lance:      "
                + ", ".join(
                    str(event_id)
                    for event_id in added[:10]
                )
            )

    mean_recall = (
        float(np.mean(overlaps))
        if overlaps
        else 0.0
    )

    min_recall = (
        float(np.min(overlaps))
        if overlaps
        else 0.0
    )

    print("\nSUMMARY")
    print("=" * 78)

    print(
        f"bucket years:       "
        f"{BUCKET_START}-{BUCKET_END}"
    )

    print(
        f"seeds:              {len(overlaps)}"
    )

    print(
        f"RRF top-N:           {TOP_N}"
    )

    print(
        f"oversample:          {OVERSAMPLE}"
    )

    print(
        f"RRF_K:               {RRF_K}"
    )

    print(
        f"nprobes:             {NPROBES}"
    )

    print(
        f"mean RRF recall:     "
        f"{mean_recall:.2%}"
    )

    print(
        f"minimum RRF recall:  "
        f"{min_recall:.2%}"
    )

    print(
        f"exact time:          "
        f"{exact_elapsed:.2f}s"
    )

    print(
        f"Lance time:          "
        f"{lance_elapsed:.2f}s"
    )

    if lance_elapsed > 0:
        print(
            f"speedup:             "
            f"{exact_elapsed / lance_elapsed:.2f}x"
        )

    total_elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"\nTotal benchmark time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()