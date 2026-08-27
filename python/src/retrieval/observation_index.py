# retrieval/observation_index.py

from abc import ABC, abstractmethod

from .models import Float32Array, SearchResult, BatchSearchResult


class ObservationIndex(ABC):
    """Backend-independent interface to observation nearest-neighbour search."""

    @abstractmethod
    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> SearchResult:
        """Return the k nearest observations for one query."""
        raise NotImplementedError


    @abstractmethod
    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
        oversample: int,
    ) -> BatchSearchResult:
        """Return the k nearest observations for each query."""
        raise NotImplementedError
