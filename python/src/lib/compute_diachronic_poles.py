from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from collections import Counter
import re

from lib.wordlist  import load_wordlist

STOPWORDS: Set[str] = load_wordlist()


def compute_centroid(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute centroid of a set of vectors."""
    return vectors.mean(axis=0, dtype=np.float32).astype(np.float32)


def compute_pc1(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute first principal component (PC1) of centered vectors."""
    centroid = compute_centroid(vectors)
    X: NDArray[np.float32] = vectors - centroid
    pca = PCA(n_components=1)
    pca.fit(X.astype(np.float32))
    pc1: NDArray[np.float32] = pca.components_[0].astype(np.float32)
    return (pc1 / np.linalg.norm(pc1)).astype(np.float32)


def project_onto_axis(
    vectors: NDArray[np.float32],
    axis: NDArray[np.float32],
    centroid: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Project vectors onto an axis (PC1)."""
    return ((vectors - centroid) @ axis).astype(np.float32)


def extract_top_words(
    contexts: List[str],
    top_n: int = 10,
    stopwords: Optional[Set[str]] = None
) -> List[Tuple[str, int]]:
    """Return the top N words by frequency from a list of context strings."""
    stopwords = stopwords or set()
    words: List[str] = [
        w.lower()
        for c in contexts
        for w in re.findall(r'\w+', c)
        if w.lower() not in stopwords
    ]
    freq: Counter[str] = Counter(words)
    return freq.most_common(top_n)


def compute_diachronic_poles(
    slice_vectors: NDArray[np.float32],
    slice_contexts: List[str],
    top_n: int = 10,
    stopwords: Optional[Set[str]] = None
) -> Dict[str, List[Tuple[str, int]]]:
    """Compute top words for positive/negative PC1 poles."""
    stopwords = stopwords or STOPWORDS
    centroid: NDArray[np.float32] = compute_centroid(slice_vectors)
    pc1: NDArray[np.float32] = compute_pc1(slice_vectors)
    scores: NDArray[np.float32] = project_onto_axis(slice_vectors, pc1, centroid)

    top_pos_idx = np.argsort(scores)[-top_n:]
    top_neg_idx = np.argsort(scores)[:top_n]

    top_pos_contexts: List[str] = [slice_contexts[i] for i in top_pos_idx]
    top_neg_contexts: List[str] = [slice_contexts[i] for i in top_neg_idx]

    return {
        'positive_pole': extract_top_words(top_pos_contexts, top_n=top_n, stopwords=stopwords),
        'negative_pole': extract_top_words(top_neg_contexts, top_n=top_n, stopwords=stopwords)
    }


if __name__ == "__main__":
    # dummy test data
    import numpy as np

    # 5 vectors of dimension 3
    vectors = np.random.rand(5, 3).astype(np.float32)
    contexts = [
        "The quick brown fox jumps over the lazy dog",
        "A stitch in time saves nine",
        "To be or not to be",
        "All that glitters is not gold",
        "Early to bed and early to rise makes a man healthy, wealthy, and wise"
    ]

    poles = compute_diachronic_poles(vectors, contexts, top_n=3)
    print("Positive pole top words:", poles["positive_pole"])
    print("Negative pole top words:", poles["negative_pole"])
