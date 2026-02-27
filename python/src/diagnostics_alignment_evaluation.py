#!/usr/bin/env python
"""
Evaluate whether orthogonal alignment improves:

1) Cross-slice anchor stability
2) PC1 directional stability for a probe term
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import lib.eebo_config as config
from lib.faiss_slices import load_slice_index, knn_search
from align import load_fasttext_vectors, load_aligned_vectors

# -------------------------
# Config
# -------------------------

REFERENCE_SLICE = "1625-1629"
PROBE_WORD = "liberty"
TOP_K = 50

ANCHORS = [
    "liberty",
    "freedom",
    "law",
    "parliament",
    "king",
    "conscience",
    "tyranny",
    "religion",
    "subject",
    "authority",
]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pca_pc1(vectors: np.ndarray) -> np.ndarray:
    """
    Compute first principal component via SVD.
    Returns unit vector.
    """
    X = vectors - vectors.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    return pc1 / np.linalg.norm(pc1)


# PART 1

def evaluate_anchor_stability() -> None:
    print("\n=== PART 1: Cross-Slice Anchor Stability ===\n")

    ref_vectors_raw = load_fasttext_vectors(REFERENCE_SLICE)
    ref_vectors_aligned = load_aligned_vectors(REFERENCE_SLICE)

    raw_scores = []
    aligned_scores = []

    for slice_range in config.SLICES:
        slice_id = f"{slice_range[0]}-{slice_range[1]}"
        if slice_id == REFERENCE_SLICE:
            continue

        raw_vectors = load_fasttext_vectors(slice_id)
        aligned_vectors = load_aligned_vectors(slice_id)

        for word in ANCHORS:
            if word not in raw_vectors or word not in ref_vectors_raw:
                continue
            raw_scores.append(
                cosine(raw_vectors[word], ref_vectors_raw[word])
            )

            if word not in aligned_vectors or word not in ref_vectors_aligned:
                continue
            aligned_scores.append(
                cosine(aligned_vectors[word], ref_vectors_aligned[word])
            )

    print(f"Unaligned mean cosine: {np.mean(raw_scores):.4f}")
    print(f"Aligned   mean cosine: {np.mean(aligned_scores):.4f}")
    print(f"Unaligned std: {np.std(raw_scores):.4f}")
    print(f"Aligned   std: {np.std(aligned_scores):.4f}")


# PART 2

def evaluate_pc1_stability() -> None:
    print("\n=== PART 2: PC1 Direction Stability ===\n")

    pc1_vectors = []

    for slice_range in config.SLICES:
        slice_id = f"{slice_range[0]}-{slice_range[1]}"
        print(f"Processing slice {slice_id}")

        index, vocab = load_slice_index(slice_range)

        # Get probe vector from aligned vectors
        aligned_vectors = load_aligned_vectors(slice_id)
        if PROBE_WORD not in aligned_vectors:
            continue

        probe_vec = aligned_vectors[PROBE_WORD]

        neighbors = knn_search(index, vocab, probe_vec, TOP_K)
        words = [w for w, _ in neighbors if w in aligned_vectors]

        vectors = np.stack([aligned_vectors[w] for w in words])
        pc1 = pca_pc1(vectors)
        pc1_vectors.append(pc1)

    # Compare PC1 across slices
    similarities = []
    for i in range(len(pc1_vectors) - 1):
        similarities.append(
            cosine(pc1_vectors[i], pc1_vectors[i + 1])
        )

    print(f"Mean cosine between adjacent PC1s: {np.mean(similarities):.4f}")
    print(f"Std deviation: {np.std(similarities):.4f}")



if __name__ == "__main__":
    evaluate_anchor_stability()
    evaluate_pc1_stability()
