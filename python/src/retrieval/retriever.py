# retrieval/retriever.py
"""
python -m retrieval.retriever
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Float32Array
from .parquet_context import ObservationContext


class ObservationRetriever(ABC):
    """Backend-independent high-level observation retrieval."""

    @abstractmethod
    def search(
        self,
        query: Float32Array,
        *,
        k: int,
    ) -> list[ObservationContext]:
        """Search observations and resolve them to human-readable context."""
        raise NotImplementedError

    @abstractmethod
    def batch_search(
        self,
        queries: Float32Array,
        *,
        k: int,
    ) -> list[list[ObservationContext]]:
        """Search multiple queries and resolve each result set."""
        raise NotImplementedError
