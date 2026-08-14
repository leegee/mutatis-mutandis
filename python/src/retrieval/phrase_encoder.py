# retrieval/phrase_encoder.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import Float32Array


class PhraseQueryEncoder(ABC):
    """Produce a query vector in the Tier 1 observation space."""

    @abstractmethod
    def encode(
        self,
        phrase: str,
    ) -> Float32Array:
        """Encode a phrase as a normalised Tier 1 query vector.

        Invariants:
            - shape is (768,)
            - dtype is float32
            - all values are finite
            - L2 norm is approximately 1
        """
        raise NotImplementedError
