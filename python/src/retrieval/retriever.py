# retrieval/retriever.py
"""
python -m retrieval.retriever
"""

from abc import ABC, abstractmethod

from .models import Float32Array, SearchSpace
from .parquet_context import ObservationContext


class ObservationRetriever(ABC):

    @abstractmethod
    def search(
        self,
        query: Float32Array,
        *,
        space: SearchSpace,
        k: int,
    ) -> list[ObservationContext]:
        """Search observations and resolve them to human-readable context."""
        raise NotImplementedError

    @abstractmethod
    def batch_search(
        self,
        queries: Float32Array,
        *,
        space: SearchSpace,
        k: int,
        oversample: int,
    ) -> list[list[ObservationContext]]:
        """Search multiple queries and resolve each result set."""
        raise NotImplementedError
