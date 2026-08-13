from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.dataset as ds

from retrieval.diskann_builder import build_diskann_index
from retrieval.diskann_observation_index import DiskANNObservationIndex


CORPUS_ROOT = Path(r"G:\corpus-out-parq\events")
YEAR = 1625
DIMENSIONS = 768

OUTPUT_DIR = Path("out/test_diskann_1625")

K = 20
QUERY_LOCAL_IDS = [0, 100, 1000, 5000, 10000]

SEARCH_COMPLEXITIES = [50, 75, 100, 150, 200, 500]


def load_observations() -> tuple[np.ndarray, np.ndarray]:
    dataset = ds.dataset(
        str(CORPUS_ROOT),
        format="parquet",
        partitioning="hive",
    )

    fragments = sorted(
        dataset.get_fragments(
            filter=(ds.field("year") == YEAR),
        ),
        key=str,
    )

    print(f"Parquet files: {len(fragments)}")

    for fragment in fragments:
        print(fragment)

    if not fragments:
        raise RuntimeError(f"No Parquet observations found for {YEAR}")

    tables = [
        fragment.to_table(
            columns=["event_id", "emb_local"],
        )
        for fragment in fragments
    ]

    import pyarrow as pa

    table = pa.concat_tables(tables)

    event_ids = np.asarray(
        table.column("event_id").to_numpy(),
        dtype=np.int64,
    )

    embeddings = np.asarray(
        table.column("emb_local").to_pylist(),
        dtype=np.float32,
    )

    print(f"events: {len(event_ids)}")
    print(f"vectors: {embeddings.shape}")
    print(f"dtype: {embeddings.dtype}")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, got shape {embeddings.shape}"
        )

    if embeddings.shape[1] != DIMENSIONS:
        raise ValueError(
            f"Expected {DIMENSIONS} dimensions, "
            f"got {embeddings.shape[1]}"
        )

    if len(event_ids) != len(embeddings):
        raise ValueError(
            "event_ids and embeddings contain different numbers "
            "of observations"
        )

    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("event IDs are not unique")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Embeddings contain non-finite values")

    norms = np.linalg.norm(embeddings, axis=1)

    print(
        "input norm range:",
        float(norms.min()),
        float(norms.max()),
    )

    return event_ids, embeddings


def exact_neighbours(
    vectors: np.ndarray,
    query_ids: list[int],
    k: int,
) -> dict[int, np.ndarray]:
    print()
    print("Computing exact ground truth...")

    result: dict[int, np.ndarray] = {}

    normalised = vectors / np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    for query_id in query_ids:
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

        result[query_id] = neighbours

    print("Ground truth complete.")

    return result


def recall_at_k(
    expected: np.ndarray,
    actual: np.ndarray,
    k: int,
) -> float:
    expected_set = set(expected[:k].tolist())
    actual_set = set(actual[:k].tolist())

    return len(expected_set & actual_set) / k


def run_recall_test(
    index: DiskANNObservationIndex,
    vectors: np.ndarray,
    event_ids: np.ndarray,
    ground_truth: dict[int, np.ndarray],
    complexity: int,
) -> tuple[float, float]:
    recalls: list[float] = []
    timings_ms: list[float] = []

    index._search_complexity = complexity

    for query_id in QUERY_LOCAL_IDS:
        start = time.perf_counter()

        result = index.search(
            vectors[query_id],
            k=K,
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        returned_event_ids = result.event_ids

        expected_local_ids = ground_truth[query_id]

        expected_event_ids = event_ids[
            expected_local_ids
        ]

        recall = recall_at_k(
            expected_event_ids,
            returned_event_ids,
            K,
        )

        recalls.append(recall)
        timings_ms.append(elapsed_ms)

    return (
        float(np.mean(recalls)),
        float(np.mean(timings_ms)),
    )


def main() -> None:
    event_ids, vectors = load_observations()

    ground_truth = exact_neighbours(
        vectors,
        QUERY_LOCAL_IDS,
        K,
    )

    if OUTPUT_DIR.exists():
        print()
        print(f"Removing existing index: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    print()
    print("Building DiskANN index...")

    event_ids_path = build_diskann_index(
        vectors,
        event_ids,
        index_directory=OUTPUT_DIR,
        dimensions=DIMENSIONS,
        complexity=100,
        graph_degree=64,
        search_memory_gb=4.0,
        build_memory_gb=8.0,
        num_threads=0,
        pq_disk_bytes=0,
        index_prefix="local",
    )

    print("DiskANN build complete.")
    print()
    print(f"Mapping: {event_ids_path}")

    print()
    print("Loading DiskANNObservationIndex...")

    index = DiskANNObservationIndex(
        index_directory=OUTPUT_DIR,
        event_ids_path=event_ids_path,
        dimensions=DIMENSIONS,
        search_complexity=100,
        beam_width=2,
        num_threads=0,
        batch_num_threads=0,
        num_nodes_to_cache=0,
        index_prefix="local",
    )

    print("Index loaded.")

    print()
    print("=" * 60)
    print("Testing individual queries")
    print("=" * 60)

    for query_id in QUERY_LOCAL_IDS:
        result = index.search(
            vectors[query_id],
            k=K,
        )

        expected_event_ids = event_ids[
            ground_truth[query_id]
        ]

        recall = recall_at_k(
            expected_event_ids,
            result.event_ids,
            K,
        )

        print()
        print(f"query local id: {query_id}")
        print(
            f"query event_id: {event_ids[query_id]}"
        )
        print(
            f"recall@{K}: {recall:.3f}"
        )
        print(
            "returned event IDs:"
        )
        print(result.event_ids)
        print(
            "distances:"
        )
        print(result.distances)

    print()
    print("=" * 60)
    print("Search-complexity sweep")
    print("=" * 60)
    print()
    print(
        f"{'complexity':>9} "
        f"{'mean recall@20':>15} "
        f"{'mean ms':>10}"
    )
    print()

    sweep_results = []

    for complexity in SEARCH_COMPLEXITIES:
        mean_recall, mean_ms = run_recall_test(
            index,
            vectors,
            event_ids,
            ground_truth,
            complexity,
        )

        sweep_results.append(
            (complexity, mean_recall, mean_ms)
        )

        print(
            f"{complexity:9d} "
            f"{mean_recall:15.3f} "
            f"{mean_ms:10.3f}"
        )

    recalls = [
        result[1]
        for result in sweep_results
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

    if np.min(recalls) < 0.99:
        raise AssertionError(
            f"DiskANN recall@{K} fell below 0.99"
        )

    print()
    print("Test complete.")


if __name__ == "__main__":
    main()
