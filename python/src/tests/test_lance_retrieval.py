from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tier1.observation_store_api import SCALES, open_observation_lookup
from tier1.parquet_observation_backend import ParquetObservationLookup
from retrieval.models import SearchSpace
from retrieval.lance_observation_index import LanceObservationIndex
from retrieval.lance_observation_index_store import LanceObservationIndexStore
from lib.corpus_config import (
    EVENTSTORE_T1_PATH,
    LANCE_INDEXES_DIR,
)

# Adjust these only if the project's paths or Lance index construction differ.
OBSERVATION_ROOT = EVENTSTORE_T1_PATH
LANCE_ROOT = LANCE_INDEXES_DIR
SEED_FORMS = ("white", "pale", "bright", "blond", "hoary")
SEEDS_PER_FORM = 3

K_VALUES = (10, 30, 60, 100, 300, 1000)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a, b) / (a_norm * b_norm))


def make_ensemble(
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    """Construct the same three-scale ensemble used by retrieval."""
    weights = np.asarray(
        [1.0 / len(SCALES)] * len(SCALES),
        dtype=np.float32,
    )

    result = None

    for weight, scale in zip(weights, SCALES):
        vector = vectors[scale]

        if result is None:
            result = weight * vector
        else:
            result += weight * vector

    if result is None:
        raise RuntimeError("No embedding scales available")

    return result.astype(np.float32)


def get_seed_event_ids(
    lookup: ParquetObservationLookup,
) -> list[int]:
    """Select a small deterministic sample of real corpus occurrences."""
    selected: list[int] = []

    for form in SEED_FORMS:
        event_ids = lookup.find_matching_event_ids([form])

        if not event_ids:
            print(f"WARNING: no occurrences found for {form!r}")
            continue

        selected.extend(event_ids[:SEEDS_PER_FORM])

    return selected


def get_seed_vectors(
    lookup: ParquetObservationLookup,
    event_ids: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Fetch all requested scale vectors while preserving event order."""
    vectors: dict[int, dict[str, np.ndarray]] = {
        event_id: {} for event_id in event_ids
    }

    for scale in SCALES:
        matrix = lookup._scale_for_ids(event_ids, scale)

        if matrix.shape[0] != len(event_ids):
            raise RuntimeError(
                f"{scale}: returned {matrix.shape[0]} vectors "
                f"for {len(event_ids)} event IDs"
            )

        for i, event_id in enumerate(event_ids):
            vectors[event_id][scale] = matrix[i]

    return vectors


def open_lance_indexes(lookup):
    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_years=lookup.available_years,
        available_scales=SCALES,
        dimensions=768,
        nprobes=20,
        model="macberth",
    )

    return store.get(
        SearchSpace(
            years=None,
            scale=tuple(SCALES),
        )
    )


def search_index(
    index,
    vector: np.ndarray,
    k: int,
):
    """Run one ANN query and return event IDs in ranked order."""
    result = index.search(vector, k=k)

    # Accommodate either a plain result object or a result containing
    # an event_id/id column depending on the current Lance wrapper.
    if hasattr(result, "event_ids"):
        ids = result.event_ids
    elif hasattr(result, "ids"):
        ids = result.ids
    else:
        raise RuntimeError(
            f"Cannot extract event IDs from search result "
            f"type {type(result)!r}"
        )

    return [int(x) for x in ids]


def benchmark_seed(
    indexes: dict[str, LanceObservationIndex],
    seed_vectors: dict[int, dict[str, np.ndarray]],
    event_id: int,
) -> dict[int, list[int]]:
    vectors = seed_vectors[event_id]

    # The current Tier 2 ensemble treats each scale as an equal contribution.
    # Keep the construction here explicit so the diagnostic remains independent
    # of the retrieval pipeline's higher-level query machinery.
    query = make_ensemble(vectors)

    print()
    print(f"Seed event_id={event_id}")

    results: dict[int, list[int]] = {}

    for k in K_VALUES:
        start = time.perf_counter()

        # Search each scale independently. This is useful because it lets us
        # see whether one particular contextual scale behaves differently.
        scale_results: dict[str, list[int]] = {}

        for scale in SCALES:
            scale_vector = vectors[scale]
            scale_results[scale] = search_index(
                indexes[scale],
                scale_vector,
                k,
            )

        elapsed = time.perf_counter() - start

        # Use the local-scale result as the primary stability series here.
        # The full per-scale results are still retained for diagnostics.
        primary = scale_results[SCALES[0]]
        results[k] = primary

        print(
            f"  k={k:4d}  "
            f"{elapsed * 1000:8.2f} ms  "
            f"local={len(scale_results['local']):4d}  "
            f"medium={len(scale_results['medium']):4d}  "
            f"broad={len(scale_results['broad']):4d}"
        )

    return results


def overlap(
    a: list[int],
    b: list[int],
) -> float:
    """Return the proportion of the smaller result contained in the larger."""
    if not a or not b:
        return 0.0

    return len(set(a) & set(b)) / min(len(a), len(b))


def print_stability(
    all_results: dict[int, dict[int, list[int]]],
) -> None:
    print()
    print("NESTED TOP-K STABILITY")
    print("=" * 70)

    for event_id, results in all_results.items():
        print(f"\nSeed event_id={event_id}")

        previous_k = K_VALUES[0]

        for k in K_VALUES[1:]:
            score = overlap(
                results[previous_k],
                results[k],
            )

            print(
                f"  {previous_k:4d} -> {k:4d}: "
                f"{score * 100:6.2f}% overlap"
            )

            previous_k = k


def main() -> None:
    print("Opening Tier 1 observation lookup...")
    lookup = open_observation_lookup(
        OBSERVATION_ROOT,
    )

    print(f"Tier 1 observations: {len(lookup):,}")

    years = lookup.available_years

    if len(years):
        print(
            f"Available years: {int(years.min())}-"
            f"{int(years.max())}"
        )

    print()
    print("TABLES")
    print("=" * 70)

    for scale in SCALES:
        print(f"{scale:>8}: expected observation population = {len(lookup):,}")

    print()
    print("Opening Lance indexes...")
    indexes = open_lance_indexes(lookup)

    print()
    print("Checking Lance indexes...")

    for scale, index in indexes.items():
        print(
            f"{scale:>8}: index opened successfully "
            f"(Tier 1 population = {len(lookup):,})"
        )

    print()
    print(
        f"Sampling seed occurrences for: "
        f"{', '.join(SEED_FORMS)}"
    )

    seed_event_ids = get_seed_event_ids(lookup)

    print(f"Selected {len(seed_event_ids)} seed occurrences.")

    if not seed_event_ids:
        raise RuntimeError("No seed occurrences found")

    print()
    print("Fetching seed vectors...")

    seed_vectors = get_seed_vectors(
        lookup,
        seed_event_ids,
    )

    print("Seed vectors loaded.")

    # Confirm that the vectors really correspond to the requested occurrences.
    for event_id in seed_event_ids:
        metadata = lookup.get_event_metadata(event_id)

        print(
            f"  {event_id}: "
            f"{metadata['token']!r} "
            f"{metadata['doc_id']} "
            f"{metadata['pub_year']}"
        )

    print()
    print("RETRIEVAL BENCHMARK")
    print("=" * 70)

    all_results: dict[int, dict[int, list[int]]] = {}

    for event_id in seed_event_ids:
        all_results[event_id] = benchmark_seed(
            indexes,
            seed_vectors,
            event_id,
        )

    print_stability(all_results)

    print()
    print("DONE")


if __name__ == "__main__":
    main()
