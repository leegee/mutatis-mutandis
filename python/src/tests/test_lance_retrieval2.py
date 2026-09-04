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

K = 1000

NPROBES_VALUES = (
    20,
    50,
    100,
    200,
    500,
)


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


def probe_overlap(
    first: list[int],
    second: list[int],
) -> float:
    """Measure candidate-set stability between two ANN searches."""
    if not first or not second:
        return 0.0

    return len(set(first) & set(second)) / min(
        len(first),
        len(second),
    )


def rank_overlap_at_k(
    first: list[int],
    second: list[int],
    k: int,
) -> float:
    """Measure overlap of the highest-ranked candidates."""
    first_set = set(first[:k])
    second_set = set(second[:k])

    if not first_set or not second_set:
        return 0.0

    return len(first_set & second_set) / min(
        len(first_set),
        len(second_set),
    )


def benchmark_scale(
    index: LanceObservationIndex,
    vector: np.ndarray,
    scale: str,
    event_id: int,
) -> None:
    print()
    print(
        f"  Scale={scale} "
        f"seed={event_id}"
    )

    results: dict[int, list[int]] = {}
    timings: dict[int, float] = {}

    for nprobes in NPROBES_VALUES:
        # The wrapper now forwards this value to Lance's vector query.
        index._nprobes = nprobes

        start = time.perf_counter()

        ann_ids = search_index(
            index,
            vector,
            K,
        )

        elapsed = time.perf_counter() - start

        results[nprobes] = ann_ids
        timings[nprobes] = elapsed

        print(
            f"    nprobes={nprobes:4d} "
            f"ANN @ {K}: "
            f"{elapsed * 1000.0:8.2f} ms "
            f"({len(ann_ids)} results) "
            f"top1={ann_ids[0] if ann_ids else None}"
        )

    print()
    print("    Candidate-set stability:")

    baseline = NPROBES_VALUES[0]

    for nprobes in NPROBES_VALUES[1:]:
        overlap = probe_overlap(
            results[baseline],
            results[nprobes],
        )

        print(
            f"      {baseline:4d} -> {nprobes:4d}: "
            f"{overlap * 100:6.2f}% "
            f"of top-{K} candidates shared"
        )

    print()
    print("    Ranked-prefix stability:")

    for prefix in (10, 30, 60, 100, 300, 1000):
        print(
            f"      k={prefix:4d}:",
            end="",
        )

        for nprobes in NPROBES_VALUES[1:]:
            overlap = rank_overlap_at_k(
                results[baseline],
                results[nprobes],
                prefix,
            )

            print(
                f"  {baseline}->{nprobes}: "
                f"{overlap * 100:6.2f}%",
                end="",
            )

        print()

    print()
    print("    Adjacent probe stability:")

    for low, high in zip(
        NPROBES_VALUES,
        NPROBES_VALUES[1:],
    ):
        overlap = probe_overlap(
            results[low],
            results[high],
        )

        timing_ratio = (
            timings[high] / timings[low]
            if timings[low] > 0
            else float("inf")
        )

        print(
            f"      {low:4d} -> {high:4d}: "
            f"overlap={overlap * 100:6.2f}% "
            f"time-ratio={timing_ratio:5.2f}x"
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
    print("LANCE NPROBES SENSITIVITY TEST")
    print("=" * 78)

    print()
    print(
        f"ANN: one top-{K} Lance query for each "
        f"nprobes value"
    )

    print(
        "No exact search is performed; this test only establishes "
        "whether changing nprobes changes ANN retrieval."
    )

    total_start = time.perf_counter()

    for event_id in seed_event_ids:
        print()
        print(
            f"Seed event_id={event_id}"
        )

        for scale in SCALES:
            benchmark_scale(
                indexes[scale],
                seed_vectors[event_id][scale],
                scale,
                event_id,
            )

    total_elapsed = (
        time.perf_counter()
        - total_start
    )

    print()
    print(
        f"Total test time: "
        f"{total_elapsed:.2f}s"
    )

    print()
    print("DONE")


if __name__ == "__main__":
    main()
