from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.dataset as ds
import diskannpy


PARQUET_ROOT = Path("G:/corpus-out-parq/events")
INDEX_DIR = Path("out/test_diskann_1625")

K_VALUES = (10, 20)
COMPLEXITIES = (50, 75, 100, 150, 200, 500)
N_QUERIES = 100
SEED = 42


files = sorted(
    PARQUET_ROOT.glob("tier1_shard*/year=1625/*.parquet")
)

print(f"Parquet files: {len(files)}")
for path in files:
    print(path)

dataset = ds.dataset(files, format="parquet")

table = dataset.to_table(
    columns=["event_id", "emb_local"]
)

event_ids = np.asarray(table["event_id"])

vectors = np.asarray(
    table["emb_local"].to_pylist(),
    dtype=np.float32,
)

print(f"events: {len(event_ids)}")
print(f"vectors: {vectors.shape}")
print(f"dtype: {vectors.dtype}")


# Unit normalisation makes L2 ranking equivalent to cosine ranking.
norms = np.linalg.norm(vectors, axis=1, keepdims=True)
vectors_norm = vectors / norms

print(
    "normalised norm range:",
    np.linalg.norm(vectors_norm, axis=1).min(),
    np.linalg.norm(vectors_norm, axis=1).max(),
)


local_event_ids = np.load(
    INDEX_DIR / "local_event_ids.npy"
)

print(f"mapping shape: {local_event_ids.shape}")
print(
    "mapping unique:",
    len(np.unique(local_event_ids)) == len(local_event_ids),
)

assert len(local_event_ids) == len(vectors)
assert np.array_equal(
    np.sort(local_event_ids),
    np.sort(event_ids),
)


index = diskannpy.StaticMemoryIndex(
    index_directory=str(INDEX_DIR),
    num_threads=0,
    initial_search_complexity=max(COMPLEXITIES),
    index_prefix="local",
    distance_metric="l2",
    vector_dtype=np.float32,
    dimensions=vectors.shape[1],
)


rng = np.random.default_rng(SEED)

query_indices = rng.choice(
    len(vectors),
    size=N_QUERIES,
    replace=False,
)

queries = vectors_norm[query_indices]


def exact_search(query_matrix: np.ndarray, k: int):
    similarities = query_matrix @ vectors_norm.T

    candidate_ids = np.argpartition(
        -similarities,
        kth=k - 1,
        axis=1,
    )[:, :k]

    candidate_scores = np.take_along_axis(
        similarities,
        candidate_ids,
        axis=1,
    )

    order = np.argsort(
        -candidate_scores,
        axis=1,
    )

    return np.take_along_axis(candidate_ids, order, axis=1)


print()
print("Computing exact ground truth...")

exact_results = {
    k: exact_search(queries, k)
    for k in K_VALUES
}

print("Ground truth complete.")
print()


print(
    f"{'complexity':>10} "
    f"{'recall@10':>10} "
    f"{'recall@20':>10} "
    f"{'mean ms':>10} "
    f"{'p95 ms':>10}"
)

print("-" * 56)


for complexity in COMPLEXITIES:
    timings = []
    recalls = {k: [] for k in K_VALUES}

    for row in range(N_QUERIES):
        query = queries[row : row + 1]

        start = perf_counter()

        response = index.batch_search(
            query,
            k_neighbors=max(K_VALUES),
            complexity=complexity,
            num_threads=0,
        )

        elapsed_ms = (perf_counter() - start) * 1000
        timings.append(elapsed_ms)

        returned = response.identifiers[0]

        for k in K_VALUES:
            exact_set = set(exact_results[k][row])
            ann_set = set(returned[:k])

            recalls[k].append(
                len(exact_set & ann_set) / k
            )

    mean_ms = np.mean(timings)
    p95_ms = np.percentile(timings, 95)

    print(
        f"{complexity:>10} "
        f"{np.mean(recalls[10]):>10.3f} "
        f"{np.mean(recalls[20]):>10.3f} "
        f"{mean_ms:>10.3f} "
        f"{p95_ms:>10.3f}"
    )


print()
print("Test complete.")
