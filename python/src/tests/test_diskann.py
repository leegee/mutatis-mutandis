from pathlib import Path
import time

import diskannpy
import numpy as np
import pyarrow.dataset as ds

from retrieval.diskann_observation_index import DiskANNObservationIndex

CORPUS_ROOT = Path(r"G:\corpus-out-parq\events")
YEAR = 1625
INDEX_DIR = Path("out/test_diskann_observation_index_1625")
INDEX_PREFIX = "local"

K = 20
SEARCH_COMPLEXITIES = (50, 75, 100, 150, 200, 500)
QUERY_LOCAL_IDS = (0, 100, 1000, 5000, 10000)


def load_observations() -> tuple[np.ndarray, np.ndarray]:
    dataset = ds.dataset(
        CORPUS_ROOT,
        format="parquet",
        partitioning="hive",
    )

    table = dataset.to_table(
        filter=ds.field("year") == YEAR,
        columns=["event_id", "emb_local"],
    )

    event_ids = np.asarray(
        table["event_id"].to_numpy(),
        dtype=np.uint64,
    )

    vectors = np.asarray(
        table["emb_local"].to_pylist(),
        dtype=np.float32,
    )

    print(f"events: {len(event_ids)}")
    print(f"vectors: {vectors.shape}")
    print(f"dtype: {vectors.dtype}")

    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D vectors, got {vectors.shape}"
        )

    if vectors.shape[1] != 768:
        raise ValueError(
            f"Expected 768-dimensional vectors, "
            f"got {vectors.shape[1]}"
        )

    if not np.isfinite(vectors).all():
        raise ValueError(
            "Vectors contain non-finite values"
        )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    print(
        "stored norm range:",
        float(norms.min()),
        float(norms.max()),
    )

    if np.any(norms == 0):
        raise ValueError(
            "Vectors contain zero-norm observations"
        )

    vectors /= norms[:, None]

    normalised_norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    print(
        "normalised norm range:",
        float(normalised_norms.min()),
        float(normalised_norms.max()),
    )

    if not np.allclose(
        normalised_norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "Normalisation failed"
        )

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError(
            "Event IDs are not unique"
        )

    return event_ids, vectors


def index_files_exist() -> bool:
    required_files = (
        f"{INDEX_PREFIX}_disk.index",
        f"{INDEX_PREFIX}_mem.index.data",
        f"{INDEX_PREFIX}_metadata.bin",
        f"{INDEX_PREFIX}_pq_compressed.bin",
        f"{INDEX_PREFIX}_pq_pivots.bin",
        f"{INDEX_PREFIX}_sample_data.bin",
        f"{INDEX_PREFIX}_sample_ids.bin",
        f"{INDEX_PREFIX}_vectors.bin",
    )

    return all(
        (INDEX_DIR / filename).is_file()
        for filename in required_files
    )


def build_index(
    vectors: np.ndarray,
) -> None:
    if index_files_exist():
        print()
        print(
            "Existing DiskANN index found; "
            "skipping index build."
        )
        print(f"index: {INDEX_DIR}")
        return

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("No complete DiskANN index found.")
    print("Building DiskANN index...")
    print(f"vectors: {vectors.shape}")
    print(f"index:   {INDEX_DIR}")

    diskannpy.build_disk_index(
        data=vectors,
        distance_metric="l2",
        index_directory=str(INDEX_DIR),
        complexity=200,
        graph_degree=64,
        search_memory_maximum=1.0,
        build_memory_maximum=2.0,
        num_threads=0,
        pq_disk_bytes=0,
        index_prefix=INDEX_PREFIX,
    )

    print("DiskANN build complete.")


def ensure_mapping(
    event_ids: np.ndarray,
) -> Path:
    mapping_path = (
        INDEX_DIR
        / f"{INDEX_PREFIX}_event_ids.npy"
    )

    if mapping_path.is_file():
        existing = np.load(
            mapping_path,
            mmap_mode="r",
        )

        if existing.ndim != 1:
            raise ValueError(
                "Existing event-ID mapping is not one-dimensional"
            )

        if existing.dtype != np.uint64:
            raise ValueError(
                "Existing event-ID mapping is not uint64"
            )

        if len(existing) != len(event_ids):
            raise ValueError(
                "Existing event-ID mapping has the wrong length"
            )

        if not np.array_equal(
            existing,
            event_ids,
        ):
            raise ValueError(
                "Existing event-ID mapping does not match "
                "the current Parquet observation order"
            )

        print(
            "Existing event-ID mapping found; "
            "skipping mapping write."
        )

        return mapping_path

    np.save(
        mapping_path,
        event_ids,
    )

    print(
        "Created event-ID mapping:",
        mapping_path,
    )

    return mapping_path


def exact_ground_truth(
    vectors: np.ndarray,
    query_ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    queries = vectors[query_ids]

    distances = np.empty(
        (len(queries), len(vectors)),
        dtype=np.float32,
    )

    for i, query in enumerate(queries):
        diff = vectors - query

        distances[i] = np.einsum(
            "ij,ij->i",
            diff,
            diff,
        )

    nearest = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )[:, :k]

    nearest_distances = np.take_along_axis(
        distances,
        nearest,
        axis=1,
    )

    return nearest, nearest_distances


def run_search_test(
    index: DiskANNObservationIndex,
    event_ids: np.ndarray,
    vectors: np.ndarray,
    query_ids: np.ndarray,
    ground_truth_local_ids: np.ndarray,
) -> tuple[float, float, float]:
    queries = vectors[query_ids]

    start = time.perf_counter()

    result = index.batch_search(
        queries,
        k=K,
    )

    elapsed = time.perf_counter() - start

    actual_event_ids = np.asarray(
        result.event_ids,
        dtype=np.uint64,
    )

    expected_event_ids = event_ids[
        ground_truth_local_ids
    ]

    overlaps = []

    for expected_row, actual_row in zip(
        expected_event_ids,
        actual_event_ids,
    ):
        overlap = len(
            set(expected_row.tolist())
            & set(actual_row.tolist())
        )

        overlaps.append(
            overlap / K
        )

    mean_recall = float(
        np.mean(overlaps)
    )

    total_ms = elapsed * 1000.0
    mean_ms = (
        elapsed
        / len(query_ids)
        * 1000.0
    )

    return (
        mean_recall,
        total_ms,
        mean_ms,
    )


def main() -> None:
    event_ids, vectors = load_observations()

    query_ids = np.asarray(
        [
            i
            for i in QUERY_LOCAL_IDS
            if i < len(vectors)
        ],
        dtype=np.int64,
    )

    print()
    print(
        "query local IDs:",
        query_ids.tolist(),
    )

    print(
        "query event IDs:",
        event_ids[query_ids].tolist(),
    )

    print()
    print("Computing exact ground truth...")

    ground_truth_local_ids, ground_truth_distances = (
        exact_ground_truth(
            vectors,
            query_ids,
            K,
        )
    )

    print("Ground truth complete.")

    build_index(vectors)

    mapping_path = ensure_mapping(
        event_ids
    )

    print()
    print("Loading DiskANNObservationIndex...")

    index = DiskANNObservationIndex(
        index_directory=INDEX_DIR,
        event_ids_path=mapping_path,
        dimensions=vectors.shape[1],
        num_threads=0,
        search_complexity=100,
        num_nodes_to_cache=0,
        index_prefix=INDEX_PREFIX,
    )

    print("ObservationIndex loaded.")

    print()
    print(
        "complexity  recall@20  mean ms/query"
    )

    results = []

    for complexity in SEARCH_COMPLEXITIES:
        index._search_complexity = complexity

        recall, total_ms, mean_ms = (
            run_search_test(
                index=index,
                vectors=vectors,
                event_ids=event_ids,
                query_ids=query_ids,
                ground_truth_local_ids=ground_truth_local_ids,
            )
        )

        results.append(
            (
                complexity,
                recall,
                mean_ms,
            )
        )

        print(
            f"{complexity:>10}"
            f"    {recall:>7.3f}"
            f"       {mean_ms:>10.3f}"
        )

    print()
    print(
        "Detailed result at complexity 100:"
    )

    index._search_complexity = 100

    result = index.batch_search(
        vectors[query_ids],
        k=K,
    )

    actual_event_ids = np.asarray(
        result.event_ids,
        dtype=np.uint64,
    )

    for row, query_id in enumerate(
        query_ids
    ):
        expected = event_ids[
            ground_truth_local_ids[row]
        ]

        actual = actual_event_ids[row]

        overlap = len(
            set(expected.tolist())
            & set(actual.tolist())
        )

        print()
        print(
            f"query local id: {query_id}"
        )

        print(
            f"query event_id:  "
            f"{event_ids[query_id]}"
        )

        print(
            f"overlap:         "
            f"{overlap}/{K}"
        )

        print(
            f"recall@{K}:       "
            f"{overlap / K:.3f}"
        )

        print("exact event IDs:")
        print(expected)

        print("DiskANN event IDs:")
        print(actual)

        print("DiskANN distances:")
        print(result.distances[row])

    recalls = [
        result[1]
        for result in results
    ]

    print()
    print("=" * 60)
    print(
        f"mean recall@{K}: "
        f"{np.mean(recalls):.3f}"
    )
    print(
        f"min  recall@{K}: "
        f"{np.min(recalls):.3f}"
    )
    print("=" * 60)

    if min(recalls) < 0.95:
        raise AssertionError(
            "DiskANN recall too low: "
            f"minimum={np.min(recalls):.3f}"
        )

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()