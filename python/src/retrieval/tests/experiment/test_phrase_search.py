# retrieval/tests/experiments/test_phrase_search.py

from __future__ import annotations

"""
Experiment: search the observation index using a MacBERTh phrase query.

Run with:

    python -m retrieval.tests.experiments.test_phrase_search
"""

from lib.corpus_config import (
    DISKANN_INDEXES_DIR,
    EVENTSTORE_T1_PATH,
)
from lib.corpus_logging import logger
from retrieval.diskann_observation_index import DiskANNObservationIndex
from retrieval.macberth_phrase_encoder import (
    DEFAULT_CARRIER,
    MacBertMeanPhraseEncoder,
)
from retrieval.observation_retriever import IndexedObservationRetriever
from retrieval.parquet_context import ParquetContext

YEAR = 1625
SCALE = "local"
DIMENSIONS = 768
K = 20

PHRASE = "preachers and teachers"


def main() -> None:
    encoder = MacBertMeanPhraseEncoder()

    logger.info("")
    logger.info("PHRASE SEARCH")
    logger.info("=" * 70)
    logger.info(f"phrase:     {PHRASE}")
    logger.info(f"encoder:    {type(encoder).__name__}")
    logger.info(f"carrier:    {DEFAULT_CARRIER}")
    logger.info(f"year:       {YEAR}")
    logger.info(f"scale:      {SCALE}")
    logger.info(f"dimensions: {DIMENSIONS}")
    logger.info(f"k:          {K}")
    logger.info("=" * 70)

    query = encoder.encode(
        PHRASE,
    )

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

    context = ParquetContext(
        EVENTSTORE_T1_PATH,
        context_before=10,
        context_after=10,
    )

    retriever = IndexedObservationRetriever(
        index=index,
        context=context,
    )

    results = retriever.search(
        query,
        k=K,
    )

    logger.info("")
    logger.info("RESULTS")
    logger.info("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        observation = result.observation

        logger.info(
            f"{rank:>2}. "
            f"{result.distance:.6f} "
            f"{result.event_id} "
            f"{observation['doc_id']} "
            f"{observation['token']!r}"
        )

        logger.info(
            f"    {result.text}"
        )

    logger.info("=" * 70)


if __name__ == "__main__":
    main()