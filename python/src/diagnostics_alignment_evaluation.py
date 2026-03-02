#!/usr/bin/env python
"""
diagnostics_alignment_evaluation.py

Evaluate orthogonal alignment quality via:

1) Static vs moving anchor cosine stability
2) PC1 neighbourhood stability for a probe word

Assumes both unaligned and aligned vectors are frozen to disk.
"""

from __future__ import annotations
import numpy as np

from slice_embedding_pipeline import (
    load_aligned_vectors,
    load_unaligned_vectors,
)
from lib.eebo_config import SLICES
from lib.eebo_logging import logger


REFERENCE_SLICE = "1625-1629"
PROBE_WORD = "liberty"
TOP_K = 50


STATIC_ANCHORS = [
    "law",
    "king",
    "subject",
    "authority",
]

MOVING_ANCHORS = [
    "liberty",
    "freedom",
    "parliament",
    "conscience",
    "tyranny",
    "religion",
]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pca_pc1(vectors: np.ndarray) -> np.ndarray:
    X = vectors - vectors.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]
    return pc1 / np.linalg.norm(pc1)



# PART 1: Anchor Stability


def evaluate_anchor_stability() -> None:
    logger.info("\n=== PART 1: Cross-Slice Anchor Stability ===\n")

    ref_raw = load_unaligned_vectors(REFERENCE_SLICE)
    ref_aligned = load_aligned_vectors(REFERENCE_SLICE)

    def collect(anchor_list: list[str]):
        raw_scores: list[float] = []
        aligned_scores: list[float] = []

        for start, end in SLICES:
            slice_id = f"{start}-{end}"
            if slice_id == REFERENCE_SLICE:
                continue

            raw_vectors = load_unaligned_vectors(slice_id)
            aligned_vectors = load_aligned_vectors(slice_id)

            for word in anchor_list:
                if word in raw_vectors and word in ref_raw:
                    raw_scores.append(
                        cosine(raw_vectors[word], ref_raw[word])
                    )

                if word in aligned_vectors and word in ref_aligned:
                    aligned_scores.append(
                        cosine(aligned_vectors[word], ref_aligned[word])
                    )

        return raw_scores, aligned_scores

    raw_static, aligned_static = collect(STATIC_ANCHORS)
    raw_moving, aligned_moving = collect(MOVING_ANCHORS)

    logger.info("STATIC ANCHORS")
    logger.info(f"  Unaligned mean: {np.mean(raw_static):.4f}")
    logger.info(f"  Aligned   mean: {np.mean(aligned_static):.4f}")
    logger.info(f"  Unaligned std:  {np.std(raw_static):.4f}")
    logger.info(f"  Aligned   std:  {np.std(aligned_static):.4f}\n")

    logger.info("MOVING ANCHORS")
    logger.info(f"  Unaligned mean: {np.mean(raw_moving):.4f}")
    logger.info(f"  Aligned   mean: {np.mean(aligned_moving):.4f}")
    logger.info(f"  Unaligned std:  {np.std(raw_moving):.4f}")
    logger.info(f"  Aligned   std:  {np.std(aligned_moving):.4f}\n")



# PART 2: PC1 Stability


def evaluate_pc1_stability(use_aligned: bool) -> None:
    label = "Aligned" if use_aligned else "Unaligned"
    logger.info(f"\n=== PART 2: PC1 Stability ({label}) ===\n")

    pc1_vectors: list[np.ndarray] = []

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        logger.info(f"Processing slice {slice_id}")

        vectors_dict = (
            load_aligned_vectors(slice_id)
            if use_aligned
            else load_unaligned_vectors(slice_id)
        )

        if PROBE_WORD not in vectors_dict:
            logger.warning(f"No probe word in {slice_id}")
            continue

        words = list(vectors_dict.keys())
        matrix = np.stack([vectors_dict[w] for w in words]).astype(np.float32)

        # Normalize entire space
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        probe_idx = words.index(PROBE_WORD)
        probe_vec = matrix[probe_idx]

        sims = matrix @ probe_vec
        sims[probe_idx] = -1.0

        k = min(TOP_K, len(words) - 1)
        nn_indices = np.argsort(-sims)[:k]

        if len(nn_indices) < 5:
            continue

        neighbor_vecs = matrix[nn_indices]
        pc1 = pca_pc1(neighbor_vecs)
        pc1_vectors.append(pc1)

    if len(pc1_vectors) < 2:
        logger.error("Not enough slices.")
        return

    similarities = [
        cosine(pc1_vectors[i], pc1_vectors[i + 1])
        for i in range(len(pc1_vectors) - 1)
    ]

    logger.info(f"Mean cosine between adjacent PC1s: {np.mean(similarities):.4f}")
    logger.info(f"Std deviation: {np.std(similarities):.4f}\n")



# MAIN

def main() -> None:
    evaluate_anchor_stability()
    evaluate_pc1_stability(use_aligned=False)
    evaluate_pc1_stability(use_aligned=True)


if __name__ == "__main__":
    main()
