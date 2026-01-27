"""
compute_synchronic_poles.py

Derive synchronic conceptual poles from a vocabulary vector space
using PCA (principal component analysis).

The first principal component (PC1) is interpreted as the main
semantic opposition structuring the slice.
"""

from __future__ import annotations
from typing import List, TypedDict, Tuple, Set
import numpy as np
from sklearn.decomposition import PCA

class PolesResult(TypedDict):
    positive_pole: List[Tuple[str, float]]
    negative_pole: List[Tuple[str, float]]
    explained_variance: float

def compute_synchronic_poles(
    word_vectors: np.ndarray,      # shape (n_words, dim)
    words: List[str],
    top_n: int = 20,
    stopwords: Set[str] | None = None,
) -> PolesResult:
    """
    Compute synchronic conceptual poles from slice vocabulary.

    Returns:
        {
            "positive_pole": List[(word, score)],
            "negative_pole": List[(word, score)],
            "explained_variance": float
        }
    """

    if stopwords is None:
        stopwords = set()

    if len(words) != word_vectors.shape[0]:
        raise ValueError("words and word_vectors must have same length")

    # --- PCA ---
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(word_vectors).flatten()

    explained_variance = float(pca.explained_variance_ratio_[0])

    # --- Pair words with scores ---
    word_scores: List[Tuple[str, float]] = [
        (w, float(score))
        for w, score in zip(words, pc1_scores, strict=True)
        if w not in stopwords
    ]

    # --- Sort by PC1 direction ---
    word_scores.sort(key=lambda x: x[1], reverse=True)

    positive_pole = word_scores[:top_n]
    negative_pole = word_scores[-top_n:]

    # Reverse negative so strongest negative first
    negative_pole = sorted(negative_pole, key=lambda x: x[1])

    return {
        "positive_pole": positive_pole,
        "negative_pole": negative_pole,
        "explained_variance": explained_variance,
    }


__all__ = ["compute_synchronic_poles"]
