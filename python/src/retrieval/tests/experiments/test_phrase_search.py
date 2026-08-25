# from __future__ import annotations

# """
# Experiment: search the observation index using a MacBERTh phrase query.

# Run with:

#     python -m retrieval.tests.experiments.test_phrase_search
# """

# from lib.corpus_logging import logger
# from lib.corpus_config import EVENTSTORE_T1_PATH
# from retrieval.diskann_observation_index_store import (
#     DiskANNObservationIndexStore,
# )
# from retrieval.macberth_phrase_encoder import (
#     DEFAULT_CARRIER,
#     MacBertMeanPhraseEncoder,
# )
# from retrieval.models import SearchSpace
# from retrieval.observation_retriever import IndexedObservationRetriever
# from retrieval.parquet_context import ParquetContext

# YEAR = None
# SCALE = None

# space = SearchSpace(
#     years=YEAR,
#     scale=SCALE,
# )

# K = 20

# # PHRASE = "preachers and teachers"

# PHRASE = "hair white as snow"


# def main() -> None:
#     encoder = MacBertMeanPhraseEncoder()

#     query = encoder.encode(
#         PHRASE,
#     )

#     index_store = DiskANNObservationIndexStore()

#     context = ParquetContext(
#         EVENTSTORE_T1_PATH,
#         context_before=10,
#         context_after=10,
#     )

#     retriever = IndexedObservationRetriever(
#         index_store=index_store,
#         context=context,
#     )

#     results = retriever.search(
#         query,
#         space=space,
#         k=K,
#     )

#     logger.info("")
#     logger.info("PHRASE SEARCH")
#     logger.info("=" * 70)
#     logger.info(f"phrase:     {PHRASE}")
#     logger.info(f"encoder:    {type(encoder).__name__}")
#     logger.info(f"carrier:    {DEFAULT_CARRIER}")
#     logger.info(f"year:       {YEAR}")
#     logger.info(f"scale:      {SCALE}")
#     logger.info(f"k:          {K}")
#     logger.info("." * 70)

#     logger.info("")
#     logger.info("RESULTS")
#     logger.info("=" * 70)

#     for rank, result in enumerate(
#         results,
#         start=1,
#     ):
#         observation = result.observation

#         logger.info(
#             f"{rank:>2}. "
#             f"{result.distance:.6f} "
#             f"{result.event_id} "
#             f"{observation['doc_id']} "
#             f"{observation['token']!r}"
#         )

#         logger.info(
#             f"    {result.text}"
#         )

#     logger.info("." * 70)


# if __name__ == "__main__":
#     main()
