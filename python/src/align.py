#!/usr/bin/env python

"""
EEBO Slice Embedding Alignment Pipeline (Reference Slice)

- All slices are aligned to a single reference slice using stable anchors
- Ensures all embeddings live in the same vector space for comparison
"""

import os
import json
from pathlib import Path
import numpy as np
from scipy.linalg import orthogonal_procrustes

from lib.eebo_logging import logger
import lib.eebo_config as config
from lib.eebo_anchor_builder import get_anchors

from train_slice_fasttext import slice_model_path
from generate_token_embeddings import generate_embeddings_per_model


def aligned_vectors_path(slice_id: str) -> Path:
    """
    Return the full path to the aligned vectors JSON for a given slice.
    """
    return config.ALIGNED_VECTORS_DIR / f"{slice_id}.json"


def save_aligned_vectors(slice_id, aligned_vectors):
    path = aligned_vectors_path(slice_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {w: vec.tolist() for w, vec in aligned_vectors.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    path = aligned_vectors_path(slice_id)
    if not path.exists():
        raise FileNotFoundError(f"Aligned vectors for slice {slice_id} not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {token: np.array(vec, dtype=np.float32) for token, vec in data.items()}


def load_fasttext_vectors(slice_id: str) -> dict[str, np.ndarray]:
    """
    Load the fastText embeddings for a given slice directly from the slice model.

    Args:
        slice_id (str): e.g. "1625-1629"

    Returns:
        dict: token -> vector (np.ndarray)
    """
    # Convert slice_id to start/end
    start, end = map(int, slice_id.split("-"))
    model_file = slice_model_path((start, end))

    if not model_file.exists():
        raise FileNotFoundError(f"FastText model for slice {slice_id} not found at {model_file}")

    embeddings = generate_embeddings_per_model(model_file)
    return embeddings


def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common_anchors = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common_anchors:
        raise ValueError("No anchor words found in both source and target embeddings")
    X = np.stack([source_vectors[w] for w in common_anchors])
    Y = np.stack([target_vectors[w] for w in common_anchors])
    R, _ = orthogonal_procrustes(X, Y)
    aligned_vectors = {w: vec @ R for w, vec in source_vectors.items()}
    return R, aligned_vectors


def align_to_reference(reference_slice_id):
    """
    Align all slices to a single reference slice.
    Returns: dict of slice_id -> aligned_vectors
    """
    anchors_dict = get_anchors()
    slice_ids = sorted(anchors_dict.keys())
    aligned_embeddings = {}

    logger.info(f"Loading reference slice vectors: {reference_slice_id}")
    ref_vectors = load_fasttext_vectors(reference_slice_id)
    ref_anchors = anchors_dict[reference_slice_id]["anchors"]

    for slice_id in slice_ids:
        logger.info(f"Processing slice: {slice_id}")
        vectors = load_fasttext_vectors(slice_id)

        if slice_id == reference_slice_id:
            aligned_vectors = vectors
            logger.info(f"Slice {slice_id} is reference; no alignment needed")
        else:
            _, aligned_vectors = orthogonal_procrustes_align(vectors, ref_vectors, ref_anchors)
            logger.info(f"Slice {slice_id} aligned to reference slice {reference_slice_id}")

        aligned_embeddings[slice_id] = aligned_vectors
        save_aligned_vectors(slice_id, aligned_vectors)

    logger.info("All slices aligned to reference successfully")
    return aligned_embeddings


if __name__ == "__main__":
    reference_slice_id = "1625-1629"
    path = aligned_vectors_path(reference_slice_id)

    if path.exists():
        logger.info(f"Loading existing aligned vectors for reference slice {reference_slice_id} from {path}")
        aligned_embeddings = {reference_slice_id: load_aligned_vectors(reference_slice_id)}
    else:
        # Align all slices and save
        aligned_embeddings = align_to_reference(reference_slice_id)
