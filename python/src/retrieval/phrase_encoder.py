# phrase_encoder.py

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PhraseQueryEncoder(ABC):
    """
    Produces a query vector in the Tier 1 observation space.

    The returned vector must already be L2 normalised so callers can
    immediately pass it to FAISS, DiskANN, or any future ANN backend.
    """

    @abstractmethod
    def encode(
        self,
        phrase: str,
    ) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            float32 array of shape (768,)
        """
