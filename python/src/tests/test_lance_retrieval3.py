# test_lance_retrieval3.py

from __future__ import annotations

import os

os.environ.setdefault("LANCE_LOG", "error")

import time

import numpy as np

from tier1.observation_store_api import (
    SCALES,
    open_observation_lookup,
)
from tier1.parquet_observation_backend import ParquetObservationLookup
from retrieval.exact_knn_search import (
    discover_shards,
    exact_knn,
)
from retrieval.models import SearchSpace
from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from lib.corpus_config import (
    EVENTSTORE_T1_PATH,
    LANCE_INDEXES_DIR,
)

OBSERVATION_ROOT = EVENTSTORE_T1_PATH
LANCE_ROOT = LANCE_INDEXES_DIR

SEED_FORMS = (
    # "white",
    # "pale",
    # "bright",
    "blond",
    # "hoary",
)

SEEDS_PER_FORM = 1

K_VALUES = (
    10,
    30,
    60,
    100,
    300,
    1000,
)

GROUND_TRUTH_K = max(K_VALUES)

NPROBES_VALUES = (
    100,
    150,
    200,
)

SEED_EVENT_IDS = {
    "white": 2558990645162150216,
    "pale": 6649491879278404957,
    "bright": 8001102461466607531,
    "blond": 8837468744104859267,
    "hoary": 6590234528837436395,
}


def get_seed_event_ids(
    lookup: ParquetObservationLookup,
) -> list[int]:
    """Select a small deterministic sample of real corpus occurrences."""
    selected: list[int] = []

    for form in SEED_FORMS:
        event_ids = lookup.find_matching_event_ids([form])

        if not event_ids:
            print(
                f"WARNING: no occurrences found for {form!r}"
            )
            continue

        selected.extend(
            int(event_id)
            for event_id in event_ids[:SEEDS_PER_FORM]
        )

    return selected


def get_seed_vectors(
    lookup: ParquetObservationLookup,
    event_ids: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Fetch all scale vectors while preserving event order."""
    vectors: dict[int, dict[str, np.ndarray]] = {
        event_id: {}
        for event_id in event_ids
    }

    for scale in SCALES:
        matrix = lookup._scale_for_ids(
            event_ids,
            scale,
        )

        if matrix.shape[0] != len(event_ids):
            raise RuntimeError(
                f"{scale}: returned {matrix.shape[0]} vectors "
                f"for {len(event_ids)} event IDs"
            )

        for index, event_id in enumerate(event_ids):
            vectors[event_id][scale] = matrix[index]

    return vectors


def open_lance_indexes(
    lookup: ParquetObservationLookup,
) -> dict[str, LanceObservationIndex]:
    store = LanceObservationIndexStore(
        LANCE_ROOT,
        available_years=lookup.available_years,
        available_scales=SCALES,
        dimensions=768,
        nprobes=NPROBES_VALUES[0],
        model="macberth",
    )

    return store.get(
        SearchSpace(
            years=None,
            scale=tuple(SCALES),
        )
    )


def search_index(
    index: LanceObservationIndex,
    vector: np.ndarray,
    k: int,
) -> list[int]:
    """Run one ANN query and return event IDs in ranked order."""
    result = index.search(
        vector,
        k=k,
    )

    return [
        int(event_id)
        for event_id in result.event_ids
    ]


def exact_search(
    vector: np.ndarray,
    shards,
    scale: str,
    k: int,
    event_id: int,
) -> tuple[list[int], float]:
    """
    Compute ground-truth neighbours by scanning the Parquet corpus.

    Failure mode:
        This scores every vector in the corpus and is deliberately expensive.
        It must remain independent of the Lance index so ANN recall is measured
        against an unrestricted reference rather than against the same index.
    """
    started = time.perf_counter()

    _scores, ids = exact_knn(
        vector.reshape(1, -1),
        shards,
        k=k,
        scale=scale,
        workers=1,
        pool="thread",
        exclude_self=True,
        query_event_ids=np.asarray(
            [event_id],
            dtype=np.int64,
        ),
    )

    elapsed = time.perf_counter() - started

    return (
        [
            int(event_id)
            for event_id in ids[0]
        ],
        elapsed,
    )


def recall_at_k(
    ann_ids: list[int],
    exact_ids: list[int],
    k: int,
) -> float:
    ann = set(ann_ids[:k])
    exact = set(exact_ids[:k])

    if not exact:
        return 0.0

    return len(ann & exact) / len(exact)


def benchmark_scale(
    index: LanceObservationIndex,
    vector: np.ndarray,
    scale: str,
    event_id: int,
    shards,
) -> dict[int, dict[str, float]]:
    print()
    print(
        f"  Scale={scale} "
        f"seed={event_id}"
    )

    print(
        f"    exact ground truth @ {GROUND_TRUTH_K}..."
    )

    exact_ids, exact_elapsed = exact_search(
        vector,
        shards,
        scale,
        GROUND_TRUTH_K,
        event_id,
    )

    print(
        f"    exact: "
        f"{exact_elapsed:.3f}s "
        f"({len(exact_ids)} results)"
    )

    results: dict[int, dict[str, float]] = {}

    for nprobes in NPROBES_VALUES:
        # The wrapper forwards this value to Lance's IVF search.
        index._nprobes = nprobes

        start = time.perf_counter()

        ann_ids = search_index(
            index,
            vector,
            k=GROUND_TRUTH_K + 1,
        )

        ann_ids = [
            candidate
            for candidate in ann_ids
            if candidate != event_id
        ][:GROUND_TRUTH_K]

        ann_elapsed = time.perf_counter() - start

        print(
            f"    nprobes={nprobes:4d} "
            f"ANN @ {GROUND_TRUTH_K}: "
            f"{ann_elapsed * 1000.0:.2f} ms "
            f"({len(ann_ids)} results)"
        )

        for k in K_VALUES:
            recall = recall_at_k(
                ann_ids,
                exact_ids,
                k,
            )

            results.setdefault(k, {})[
                f"recall_{nprobes}"
            ] = recall

            print(
                f"      k={k:4d} "
                f"recall={recall * 100:6.2f}%"
            )

    return results


def print_summary(
    all_results: dict[
        str,
        dict[int, dict[int, dict[str, float]]],
    ],
) -> None:
    print()
    print("RECALL SUMMARY")
    print("=" * 100)

    for scale in SCALES:
        scale_results = all_results[scale]

        print()
        print(f"{scale.upper()}")

        for nprobes in NPROBES_VALUES:
            print()
            print(f"nprobes={nprobes}")
            print(
                f"{'k':>6} "
                f"{'mean recall':>15} "
                f"{'min recall':>15}"
            )
            print("-" * 42)

            for k in K_VALUES:
                rows = [
                    result[k][f"recall_{nprobes}"]
                    for result in scale_results.values()
                ]

                mean_recall = float(
                    np.mean(rows)
                )

                min_recall = float(
                    np.min(rows)
                )

                print(
                    f"{k:6d} "
                    f"{mean_recall * 100:14.2f}% "
                    f"{min_recall * 100:14.2f}%"
                )


def main() -> None:
    print("Opening Tier 1 observation lookup...")

    lookup = open_observation_lookup(
        OBSERVATION_ROOT,
    )

    observation_count = len(lookup)

    print(
        f"Tier 1 observations: "
        f"{observation_count:,}"
    )

    years = lookup.available_years

    if len(years):
        print(
            f"Available years: "
            f"{int(years.min())}-"
            f"{int(years.max())}"
        )

    print()
    print("Opening Lance indexes...")

    indexes = open_lance_indexes(
        lookup,
    )

    print()
    print("Opening exact-search Parquet shards...")

    shards = discover_shards(
        OBSERVATION_ROOT,
    )

    print(
        f"Parquet shards: "
        f"{len(shards):,}"
    )

    print()
    print("Checking Lance indexes...")

    for scale in SCALES:
        print(
            f"{scale:>8}: index opened successfully "
            f"(Tier 1 population = "
            f"{observation_count:,})"
        )

    print()
    print(
        "Sampling seed occurrences for: "
        f"{', '.join(SEED_FORMS)}"
    )

    print(
        f"Seeds per form: {SEEDS_PER_FORM}"
    )

    seed_event_ids = get_seed_event_ids(
        lookup,
    )

    print(
        f"Selected "
        f"{len(seed_event_ids)} "
        f"seed occurrences."
    )

    if not seed_event_ids:
        raise RuntimeError(
            "No seed occurrences found"
        )

    print()
    print("Fetching seed vectors...")

    seed_vectors = get_seed_vectors(
        lookup,
        seed_event_ids,
    )

    print("Seed vectors loaded.")

    for event_id in seed_event_ids:
        metadata = lookup.get_event_metadata(
            event_id,
        )

        print(
            f"  {event_id}: "
            f"{metadata['token']!r} "
            f"{metadata['doc_id']} "
            f"{metadata['pub_year']}"
        )

    print()
    print("EXACT VS ANN RECALL BENCHMARK")
    print("=" * 78)

    print()
    print(
        f"Ground truth: independent exact top-{GROUND_TRUTH_K} "
        f"Parquet search for each sampled seed and scale"
    )

    print(
        f"ANN: one top-{GROUND_TRUTH_K + 1} Lance query for each "
        f"nprobes value; the seed occurrence is removed before "
        f"evaluating the resulting top-{GROUND_TRUTH_K}; smaller "
        f"k values are evaluated from the same ranked result"
    )

    all_results: dict[
        str,
        dict[int, dict[int, dict[str, float]]],
    ] = {
        scale: {}
        for scale in SCALES
    }

    total_start = time.perf_counter()

    for event_id in seed_event_ids:
        print()
        print(
            f"Seed event_id={event_id}"
        )

        for scale in SCALES:
            result = benchmark_scale(
                indexes[scale],
                seed_vectors[event_id][scale],
                scale,
                event_id,
                shards,
            )

            all_results[scale][event_id] = result

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print_summary(
        all_results,
    )

    print()
    print(
        f"Total benchmark time: "
        f"{total_elapsed / 60.0:.2f} minutes"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
