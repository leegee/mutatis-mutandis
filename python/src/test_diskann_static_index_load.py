from pathlib import Path

import diskannpy
import numpy as np


INDEX_DIRECTORY = Path("out/test_diskann_diagnostic")
DIMENSIONS = 8
INDEX_PREFIX = "diagnostic"

QUERY_LOCAL_ID = 100
K_VALUES = [1, 2, 5, 10, 20]


def load_vectors() -> np.ndarray:
    path = INDEX_DIRECTORY / f"{INDEX_PREFIX}_vectors.bin"

    # The DiskANN vector file has a small binary header:
    # first uint32 = number of points, second uint32 = dimensions.
    with path.open("rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=2)
        if header.shape != (2,):
            raise RuntimeError("Could not read vector-file header")

        n_points, dimensions = map(int, header)

        if dimensions != DIMENSIONS:
            raise ValueError(
                f"Expected {DIMENSIONS} dimensions, got {dimensions}"
            )

        vectors = np.fromfile(
            f,
            dtype=np.float32,
            count=n_points * dimensions,
        )

    vectors = vectors.reshape(n_points, dimensions)

    if not np.isfinite(vectors).all():
        raise ValueError("DiskANN vector file contains non-finite values")

    return vectors


def exact_neighbours(
    vectors: np.ndarray,
    query_id: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    normalised = vectors / np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    query = normalised[query_id]

    distances = 2.0 - 2.0 * (
        normalised @ query
    )

    neighbours = np.argpartition(
        distances,
        k - 1,
    )[:k]

    neighbours = neighbours[
        np.argsort(distances[neighbours])
    ]

    return neighbours, distances[neighbours]


def main() -> None:
    print(f"index directory: {INDEX_DIRECTORY}")
    print(f"dimensions: {DIMENSIONS}")
    print(f"query local ID: {QUERY_LOCAL_ID}")

    vectors = load_vectors()

    print(f"vectors: {vectors.shape}")
    print(f"dtype: {vectors.dtype}")

    query = vectors[QUERY_LOCAL_ID]

    norm = np.linalg.norm(query)

    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Query vector has invalid norm")

    query = query / norm

    print(f"query norm: {np.linalg.norm(query)}")

    exact_ids, exact_distances = exact_neighbours(
        vectors,
        QUERY_LOCAL_ID,
        max(K_VALUES),
    )

    print()
    print("=" * 60)
    print("EXACT GROUND TRUTH")
    print("=" * 60)

    print("local IDs:")
    print(exact_ids)

    print("distances:")
    print(exact_distances)

    print()
    print("=" * 60)
    print("LOADING NATIVE STATIC DISK INDEX")
    print("=" * 60)

    # num_threads=0 is intentional. Explicit num_threads=1 was observed
    # to hang during native StaticDiskIndex construction.
    index = diskannpy.StaticDiskIndex(
        index_directory=str(INDEX_DIRECTORY),
        num_threads=0,
        num_nodes_to_cache=0,
        cache_mechanism=0,
        distance_metric="l2",
        vector_dtype=np.float32,
        dimensions=DIMENSIONS,
        index_prefix=INDEX_PREFIX,
    )

    print("Index loaded.", flush=True)

    print()
    print("=" * 60)
    print("NATIVE DISKANN SEARCH")
    print("=" * 60)

    for k in K_VALUES:
        print()
        print(f"k={k}")

        response = index.search(
            query,
            k_neighbors=k,
            complexity=100,
            beam_width=2,
        )

        identifiers = np.asarray(
            response.identifiers,
            dtype=np.int64,
        )

        distances = np.asarray(
            response.distances,
            dtype=np.float32,
        )

        unique_count = len(np.unique(identifiers))

        expected = exact_ids[:k]

        recall = (
            len(set(expected.tolist()) & set(identifiers.tolist()))
            / k
        )

        ordered = bool(
            np.all(distances[:-1] <= distances[1:])
        )

        print("identifiers:")
        print(identifiers)

        print("distances:")
        print(distances)

        print(
            f"unique IDs: {unique_count}/{len(identifiers)}"
        )

        print(
            f"recall@{k}: {recall:.3f}"
        )

        print(
            f"distances ordered: {ordered}"
        )

        if unique_count != len(identifiers):
            print(
                "WARNING: native DiskANN returned duplicate IDs"
            )

        if not ordered:
            print(
                "WARNING: native DiskANN distances are not ordered"
            )


if __name__ == "__main__":
    main()