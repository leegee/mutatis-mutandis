from __future__ import annotations

from .models import Float32Array, SearchSpace
from .observation_index_store import ObservationIndexStore
from .parquet_context import ObservationContext, ParquetContext
from .retriever import ObservationRetriever


class IndexedObservationRetriever(ObservationRetriever):
    """Resolve searches in selected observation spaces into corpus context."""

    def __init__(
        self,
        index_store: ObservationIndexStore,
        context: ParquetContext,
    ) -> None:
        self._index_store = index_store
        self._context = context

    def search(
        self,
        query: Float32Array,
        *,
        space: SearchSpace,
        k: int,
    ) -> list[ObservationContext]:
        indexes = self._index_store.get(space)

        if len(indexes) != 1:
            raise NotImplementedError(
                "multi-index search is not implemented"
            )

        result = indexes[0].search(
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
        space: SearchSpace,
        k: int,
    ) -> list[list[ObservationContext]]:
        indexes = self._index_store.get(space)

        if len(indexes) != 1:
            raise NotImplementedError(
                "multi-index search is not implemented"
            )

        results = indexes[0].batch_search(
            queries,
            k=k,
        )

        return [
            self._context.get_many(
                search_result,
            )
            for search_result in results
        ]
