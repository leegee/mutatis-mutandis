from __future__ import annotations

import numpy as np
import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from retrieval.exact_knn_search import discover_shards, exact_knn
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace


SEED_EVENT_ID = 8837468744104859267
SCALE = "broad"
K = 10
DIMENSIONS = 768

PARQUET_STORE = LANCE_INDEXES_DIR.parent.parent / "events"


def get_query_vector(
    index,
    event_id: int,
) -> np.ndarray:
    vector = np.asarray(
        index.reconstruct(event_id),
        dtype=np.float32,
    )

    if vector.shape != (DIMENSIONS,):
        raise ValueError(
            f"Expected query vector shape "
            f"({DIMENSIONS},), got {vector.shape}"
        )

    norm = np.linalg.norm(vector)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError(
            f"Invalid query vector for event {event_id}"
        )

    return vector / norm


def lance_bypass_ids(
    table,
    query_vector: np.ndarray,
    k: int,
) -> np.ndarray:
    rows = (
        table.search(
            query_vector,
            vector_column_name="vector",
        )
        .limit(k + 1)
        .bypass_vector_index()
        .select(["event_id", "_distance"])
        .to_list()
    )

    ids = np.asarray(
        [row["event_id"] for row in rows],
        dtype=np.uint64,
    )

    ids = ids[
        ids != np.uint64(SEED_EVENT_ID)
    ]

    return ids[:k]


def main() -> None:
    print(f"seed={SEED_EVENT_ID}")
    print(f"scale={SCALE}")
    print(f"k={K}")
    print(f"parquet store={PARQUET_STORE}")

    if not PARQUET_STORE.exists():
        raise FileNotFoundError(
            f"Parquet store not found: {PARQUET_STORE}"
        )

    store = LanceObservationIndexStore(
        LANCE_INDEXES_DIR,
        available_years=tuple(range(1476, 1921)),
        available_scales=(SCALE,),
    )

    indexes = store.get(
        SearchSpace(
            years=None,
            scale=(SCALE,),
        )
    )

    index = indexes[SCALE]

    query_vector = get_query_vector(
        index,
        SEED_EVENT_ID,
    )

    print("query vector reconstructed from Lance")

    print("\ndiscovering Parquet shards...")

    shards = discover_shards(PARQUET_STORE)

    print(
        f"discovered {len(shards)} shards "
        f"({sum(s.n_rows for s in shards):,} rows)"
    )

    print("\nrunning independent Parquet exact search...")

    # exact_knn returns (scores, ids), not (ids, scores).
    exact_scores, exact_ids = exact_knn(
        query_vector[None, :],
        shards,
        k=K,
        scale=SCALE,
        workers=1,
        pool="process",
        exclude_self=True,
        query_event_ids=np.asarray(
            [SEED_EVENT_ID],
            dtype=np.int64,
        ),
    )

    exact_scores = np.asarray(
        exact_scores[0],
        dtype=np.float32,
    )

    exact_ids = np.asarray(
        exact_ids[0],
        dtype=np.int64,
    )

    print("\nParquet exact:")

    for rank, (event_id, score) in enumerate(
        zip(exact_ids, exact_scores),
        start=1,
    ):
        print(
            f"{rank:2d} "
            f"{event_id} "
            f"score={score:.8f}"
        )

    print("\nrunning Lance exhaustive search...")

    db = lancedb.connect(
        str(LANCE_INDEXES_DIR)
    )

    table = db.open_table(SCALE)

    bypass_ids = lance_bypass_ids(
        table,
        query_vector,
        K,
    )

    print("\nLance bypass:")

    for rank, event_id in enumerate(
        bypass_ids,
        start=1,
    ):
        print(
            f"{rank:2d} {event_id}"
        )

    exact_set = {
        int(event_id)
        for event_id in exact_ids
        if event_id >= 0
    }

    bypass_set = {
        int(event_id)
        for event_id in bypass_ids
    }

    overlap = exact_set & bypass_set

    print("\nComparison:")
    print(
        f"Parquet exact results: {len(exact_set)}"
    )
    print(
        f"Lance bypass results:  {len(bypass_set)}"
    )
    print(
        f"overlap:               {len(overlap)}/{K}"
    )
    print(
        f"recall:                {len(overlap) / K:.2%}"
    )

    if exact_ids.size:
        print(
            f"Parquet exact top-1:   "
            f"{int(exact_ids[0])}"
        )

    if bypass_ids.size:
        print(
            f"Lance bypass top-1:   "
            f"{int(bypass_ids[0])}"
        )


if __name__ == "__main__":
    main()