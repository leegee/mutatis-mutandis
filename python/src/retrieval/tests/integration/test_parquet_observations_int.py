from __future__ import annotations

"""
pytest src/retrieval/tests/integration/test_parquet_observations_int.py -v -s
"""

import numpy as np

from lib.corpus_config import (
    DISKANN_INDEXES_DIR,
    EVENTSTORE_T1_PATH,
)
from lib.corpus_logging import logger
from retrieval.diskann_observation_index import DiskANNObservationIndex
from retrieval.parquet_context import ParquetContext
from retrieval.parquet_embeddings import load_embeddings
from retrieval.parquet_observations import ParquetObservationStore

YEAR = 1625
SCALE = "local"
DIMENSIONS = 768
K = 20


def test_diskann_results_resolve_to_parquet_observations() -> None:
    event_ids, vectors = load_embeddings(
        EVENTSTORE_T1_PATH,
        year_start=YEAR,
        year_start=YEAR+1,
        scale=SCALE,
        dimensions=DIMENSIONS,
    )

    assert len(event_ids) == len(vectors)
    assert len(event_ids) > 0

    index_directory = (
        DISKANN_INDEXES_DIR
        / f"year={YEAR}"
        / SCALE
    )

    index = DiskANNObservationIndex(
        index_directory=index_directory,
        event_ids_path=(
            index_directory
            / f"{SCALE}_event_ids.npy"
        ),
        dimensions=DIMENSIONS,
        num_threads=0,
        search_complexity=100,
        beam_width=2,
        num_nodes_to_cache=0,
        index_prefix=SCALE,
    )

    result = index.search(
        vectors[0],
        k=K,
    )

    assert len(result.event_ids) == K
    assert len(result.distances) == K

    assert result.event_ids.dtype == np.uint64
    assert result.distances.dtype == np.float32
    assert np.isfinite(result.distances).all()

    observation_store = ParquetObservationStore(
        EVENTSTORE_T1_PATH,
    )

    observations = observation_store.get_many_ordered(
        result.event_ids,
    )

    assert len(observations) == K
    assert all(
        observation is not None
        for observation in observations
    )

    for event_id, observation in zip(
        result.event_ids,
        observations,
    ):
        assert observation is not None
        assert int(observation["event_id"]) == int(event_id)

    logger.info("")
    logger.info("DISKANN → PARQUET OBSERVATION LOOKUP")
    logger.info("")
    logger.info(f"year:       {YEAR}")
    logger.info(f"scale:      {SCALE}")
    logger.info(f"dimensions: {DIMENSIONS}")
    logger.info(f"k:          {K}")
    logger.info("")

    for rank, (
        event_id,
        distance,
        observation,
    ) in enumerate(
        zip(
            result.event_ids,
            result.distances,
            observations,
        ),
        start=1,
    ):
        assert observation is not None

        logger.info(
            f"{rank:>2}. "
            f"{float(distance):.6f} "
            f"{int(event_id)} "
            f"{observation}"
        )

    context = ParquetContext(
        EVENTSTORE_T1_PATH,
        context_before=10,
        context_after=10,
    )

    first_context = context.get_many(
        result,
    )[0]

    first_observation = first_context.observation

    assert first_context.event_id == int(result.event_ids[0])
    assert first_observation["doc_id"] == observations[0]["doc_id"]
    assert first_observation["token"] == observations[0]["token"]
    assert first_observation["token_idx"] == observations[0]["token_idx"]

    logger.info("")
    logger.info("CONTEXT")
    logger.info("=" * 70)
    logger.info(f"doc:       {first_observation['doc_id']}")
    logger.info(f"token:     {first_observation['token']}")
    logger.info(f"token_idx: {first_observation['token_idx']}")
    logger.info("-" * 70)
    logger.info(first_context.text)
    logger.info("=" * 70)

