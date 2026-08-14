# observation_retriever.py

from __future__ import annotations

from .models import Float32Array
from .observation_index import ObservationIndex
from .parquet_context import ObservationContext, ParquetContext
from .retriever import ObservationRetriever


class IndexedObservationRetriever(ObservationRetriever):
    """Resolve ANN searches into Parquet-backed observation context."""

    def __init__(
        self,
        index: ObservationIndex,
        context: ParquetContext,
    ) -> None:
        self._index = index
        self._context = context

    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> list[ObservationContext]:
        result = self._index.search(
            query,
            k=k,
        )

        return self._context.get_many(
            result,
        )

    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
    ) -> list[list[ObservationContext]]:
        result = self._index.batch_search(
            queries,
            k=k,
        )

        return [
            self._context.get_many(
                search_result,
            )
            for search_result in result
        ]
