#!/usr/bin/env python
"""
End-to-end test of arbitrary phrase queries against the Tier 1.5
Parquet/DiskANN observation index.

The phrase does not need to occur in the corpus. It is encoded directly
into the Tier 1 observation embedding space, searched against DiskANN,
and the returned event IDs are resolved through the Parquet observation
layer.

This deliberately tests one phrase at a time and has no CLI arguments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import lib.corpus_config as config
from retrieval.diskann_observation_index import DiskANNObservationIndex
from retrieval.parquet_embeddings import load_embeddings
from retrieval.macberth_phrase_encoder import MacBertMeanPhraseEncoder


YEAR = 1625
SCALE = "local"
DIMENSIONS = 768
K = 20
SEARCH_COMPLEXITY = 100

PHRASES = (
    "extreme whiteness",
    "unnatural pallor",
)


def load_index() -> DiskANNObservationIndex:
    index_directory = (
        config.DISKANN_INDEXES_DIR
        / f"year={YEAR}"
        / SCALE
    )

    event_ids_path = (
        index_directory
        / f"{SCALE}_event_ids.npy"
    )

    if not index_directory.exists():
        raise RuntimeError(
            f"DiskANN index does not exist: {index_directory}"
        )

    if not event_ids_path.exists():
        raise RuntimeError(
            f"DiskANN event-ID mapping does not exist: "
            f"{event_ids_path}"
        )

    return DiskANNObservationIndex(
        index_directory=index_directory,
        event_ids_path=event_ids_path,
        dimensions=DIMENSIONS,
        num_threads=0,
        num_nodes_to_cache=0,
        index_prefix=SCALE,
    )


def load_observation_metadata() -> dict[int, dict]:
    """
    Build the small event-ID -> observation metadata mapping needed for
    displaying the search results.

    The vector matrix is loaded here only because this is an integration
    test; the production query path should not need to materialise it.
    """
    event_ids, _ = load_embeddings(
        config.EVENTSTORE_T1_PATH,
        year=YEAR,
        scale=SCALE,
        dimensions=DIMENSIONS,
    )

    return {
        int(event_id): {
            "event_id": int(event_id),
        }
        for event_id in event_ids
    }


def main() -> None:
    print("=" * 80)
    print("PHRASE → DISKANN INTEGRATION TEST")
    print("=" * 80)
    print(f"year:       {YEAR}")
    print(f"scale:      {SCALE}")
    print(f"dimensions: {DIMENSIONS}")
    print(f"k:          {K}")
    print()

    print("Loading phrase encoder...")
    encoder = MacBertMeanPhraseEncoder()

    print("Loading DiskANN...")
    index = load_index()

    index._search_complexity = SEARCH_COMPLEXITY

    print("Loading observation IDs...")
    metadata = load_observation_metadata()

    for phrase in PHRASES:
        print()
        print("=" * 80)
        print(f"PHRASE: {phrase!r}")
        print("=" * 80)

        query = encoder.encode(phrase)

        if query.shape != (DIMENSIONS,):
            raise AssertionError(
                f"Expected query shape ({DIMENSIONS},), "
                f"got {query.shape}"
            )

        if query.dtype != np.float32:
            raise AssertionError(
                f"Expected float32 query, got {query.dtype}"
            )

        norm = np.linalg.norm(query)

        if not np.isclose(norm, 1.0, atol=1e-5):
            raise AssertionError(
                f"Query is not normalised: norm={norm}"
            )

        result = index.batch_search(
            query.reshape(1, -1),
            k=K,
        )

        event_ids = np.asarray(
            result.event_ids[0],
            dtype=np.uint64,
        )

        distances = np.asarray(
            result.distances[0],
            dtype=np.float32,
        )

        print()
        print(
            f"{'rank':>4} "
            f"{'distance':>12} "
            f"{'event_id':>22}"
        )
        print("-" * 45)

        for rank, (event_id, distance) in enumerate(
            zip(event_ids, distances),
            start=1,
        ):
            event_id = int(event_id)

            if event_id not in metadata:
                raise AssertionError(
                    f"DiskANN returned unknown event ID: {event_id}"
                )

            print(
                f"{rank:>4} "
                f"{distance:>12.6f} "
                f"{event_id:>22}"
            )


if __name__ == "__main__":
    main()
