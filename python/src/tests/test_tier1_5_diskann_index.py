#!/usr/bin/env python
"""
Test an existing Tier 1.5 DiskANN index against brute-force ground truth.

The index must already have been built by the Tier 1.5 builder. This test
does not build, modify, or delete the index.

Parquet is the source of truth for vectors and event IDs. DiskANN supplies
the approximate-neighbour results.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import lib.corpus_config as config
from retrieval.diskann_observation_index import DiskANNObservationIndex
from retrieval.parquet_embeddings import load_embeddings

YEAR = 1625
SCALE = "local"
DIMENSIONS = 768
K = 20

INDEX_DIRECTORY = Path("out/test_diskann_observation_index_1625")
INDEX_PREFIX = SCALE
EVENT_IDS_PATH = INDEX_DIRECTORY / f"{INDEX_PREFIX}_event_ids.npy"

SEARCH_COMPLEXITIES = (50, 75, 100, 150, 200, 500)
QUERY_LOCAL_IDS = (0, 100, 1000, 5000, 10000)


def load_observations() -> tuple[np.ndarray, np.ndarray]:
    event_ids, vectors = load_embeddings(
        config.EVENTSTORE_T1_PATH,
        year_start=YEAR,
        year_end=YEAR+1,
        scale=SCALE,
        dimensions=DIMENSIONS,
    )

    print(f"events:  {len(event_ids)}")
    print(f"vectors: {vectors.shape}")
    print(f"dtype:   {vectors.dtype}")

    if vectors.ndim != 2 or vectors.shape[1] != DIMENSIONS:
        raise ValueError(
            f"Expected ({len(event_ids)}, {DIMENSIONS}) vectors, "
            f"got {vectors.shape}"
        )

    if len(event_ids) != len(vectors):
        raise ValueError("Event-ID and vector counts differ")

    if not np.isfinite(vectors).all():
        raise ValueError("Vectors contain non-finite values")

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("Event IDs are not unique")

    norms = np.linalg.norm(vectors, axis=1)

    if np.any(norms == 0):
        raise ValueError("Vectors contain zero-norm observations")

    print(
        "stored norm range:",
        float(norms.min()),
        float(norms.max()),
    )

    vectors = vectors / norms[:, None]

    return event_ids, vectors


def exact_ground_truth(
    vectors: np.ndarray,
    query_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    nearest = np.empty(
        (len(query_ids), k),
        dtype=np.int64,
    )

    for row, query_id in enumerate(query_ids):
        query = vectors[query_id]
        diff = vectors - query
        distances = np.einsum(
            "ij,ij->i",
            diff,
            diff,
        )

        nearest[row] = np.argsort(
            distances,
            kind="stable",
        )[:k]

    return nearest


def recall_at_k(
    expected_event_ids: np.ndarray,
    actual_event_ids: np.ndarray,
) -> float:
    recalls = []

    for expected, actual in zip(
        expected_event_ids,
        actual_event_ids,
    ):
        recalls.append(
            len(
                set(expected.tolist())
                & set(actual.tolist())
            )
            / K
        )

    return float(np.mean(recalls))


def main() -> None:
    if not INDEX_DIRECTORY.exists():
        raise RuntimeError(
            f"DiskANN index does not exist: {INDEX_DIRECTORY}"
        )

    if not EVENT_IDS_PATH.exists():
        raise RuntimeError(
            f"Event-ID mapping does not exist: {EVENT_IDS_PATH}"
        )

    print("=" * 70)
    print("TIER 1.5 DISKANN INTEGRATION TEST")
    print("=" * 70)
    print(f"year:       {YEAR}")
    print(f"scale:      {SCALE}")
    print(f"dimensions: {DIMENSIONS}")
    print(f"k:          {K}")
    print(f"index:      {INDEX_DIRECTORY}")
    print()

    print("=" * 70)
    print("LOADING PARQUET OBSERVATIONS")
    print("=" * 70)

    event_ids, vectors = load_observations()

    query_ids = np.asarray(
        [
            query_id
            for query_id in QUERY_LOCAL_IDS
            if query_id < len(vectors)
        ],
        dtype=np.int64,
    )

    if len(query_ids) == 0:
        raise RuntimeError("None of the configured query IDs exist")

    print("query local IDs:", query_ids.tolist())
    print("query event IDs:", event_ids[query_ids].tolist())

    print()
    print("=" * 70)
    print("COMPUTING EXACT GROUND TRUTH")
    print("=" * 70)

    ground_truth_local_ids = exact_ground_truth(
        vectors,
        query_ids,
        K,
    )

    ground_truth_event_ids = event_ids[
        ground_truth_local_ids
    ]

    print("Ground truth complete.")

    print()
    print("=" * 70)
    print("LOADING EXISTING TIER 1.5 DISKANN INDEX")
    print("=" * 70)

    index = DiskANNObservationIndex(
        index_directory=INDEX_DIRECTORY,
        event_ids_path=EVENT_IDS_PATH,
        dimensions=DIMENSIONS,
        num_threads=0,
        num_nodes_to_cache=0,
        index_prefix=INDEX_PREFIX,
    )

    print("ObservationIndex loaded.")

    print()
    print("=" * 70)
    print("SEARCH COMPLEXITY TEST")
    print("=" * 70)
    print()
    print(
        f"{'complexity':>10}"
        f" {'recall@20':>12}"
        f" {'mean ms/query':>16}"
    )

    results: list[tuple[int, float, float]] = []

    for complexity in SEARCH_COMPLEXITIES:
        index._search_complexity = complexity

        start = time.perf_counter()

        result = index.batch_search(
            vectors[query_ids],
            k=K,
        )

        elapsed = time.perf_counter() - start

        actual_event_ids = np.asarray(
            result.event_ids,
            dtype=np.uint64,
        )

        mean_recall = recall_at_k(
            ground_truth_event_ids,
            actual_event_ids,
        )

        mean_ms = (
            elapsed
            / len(query_ids)
            * 1000.0
        )

        results.append(
            (complexity, mean_recall, mean_ms)
        )

        print(
            f"{complexity:>10}"
            f" {mean_recall:>12.3f}"
            f" {mean_ms:>16.3f}"
        )

    print()
    print("=" * 70)
    print("DETAILED RESULT AT COMPLEXITY 100")
    print("=" * 70)

    index._search_complexity = 100

    result = index.batch_search(
        vectors[query_ids],
        k=K,
    )

    actual_event_ids = np.asarray(
        result.event_ids,
        dtype=np.uint64,
    )

    for row, query_id in enumerate(query_ids):
        expected = ground_truth_event_ids[row]
        actual = actual_event_ids[row]

        overlap = len(
            set(expected.tolist())
            & set(actual.tolist())
        )

        distances = np.asarray(
            result.distances[row],
            dtype=np.float32,
        )

        print()
        print(f"query local id: {query_id}")
        print(f"query event_id:  {event_ids[query_id]}")
        print(f"overlap:         {overlap}/{K}")
        print(f"recall@{K}:       {overlap / K:.3f}")

        print("exact event IDs:")
        print(expected)

        print("DiskANN event IDs:")
        print(actual)

        print("DiskANN distances:")
        print(distances)

        if len(np.unique(actual)) != len(actual):
            print("WARNING: duplicate IDs returned")

        if np.all(distances == 0):
            print("WARNING: every returned distance is zero")

    recalls = [result[1] for result in results]

    print()
    print("=" * 70)
    print(f"mean recall@{K}: {np.mean(recalls):.3f}")
    print(f"min  recall@{K}:  {np.min(recalls):.3f}")
    print("=" * 70)

    if min(recalls) < 0.95:
        raise AssertionError(
            "Tier 1.5 DiskANN recall too low: "
            f"minimum={min(recalls):.3f}"
        )

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()
