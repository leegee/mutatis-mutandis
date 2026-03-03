#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (aligned or unaligned) and build FAISS indexes.

- Embeddings: train or load fastText models per slice, or compute aligned embeddings
- FAISS indices: stored per slice in separate subdirs for aligned vs unaligned
- Provides methods to add and search the index
- Env var `USE_ALIGNED_FASTTEXT_VECTORS` and flag `--aligned`
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Callable, cast

import fasttext
import numpy as np
import faiss
from scipy.linalg import orthogonal_procrustes

from lib.eebo_logging import logger
from lib.eebo_config import (
    FASTTEXT_SLICE_MODEL_DIR, SLICES, SLICES_DIR,
    FAISS_INDEX_DIR, ALIGNED_VECTORS_DIR, FASTTEXT_PARAMS
)
from lib.eebo_anchor_builder import get_anchors


def unaligned_vectors_path(slice_id: str) -> Path:
    return ALIGNED_VECTORS_DIR / "unaligned" / f"{slice_id}.npz"


def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    return FASTTEXT_SLICE_MODEL_DIR / f"slice_{start}_{end}.bin"


def faiss_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = FAISS_INDEX_DIR / ("aligned" if aligned else "unaligned")
    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.faiss"

def vocab_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = FAISS_INDEX_DIR / ("aligned" if aligned else "unaligned")
    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.vocab"


def aligned_vectors_path(slice_id: str) -> Path:
    return ALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def save_aligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    ALIGNED_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    path_str = str(aligned_vectors_path(slice_id))
    np.savez(path_str, data=np.array(embeddings, dtype=object))


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    path = aligned_vectors_path(slice_id)
    if not path.exists():
        raise FileNotFoundError(f"Aligned vectors missing: {path}")
    with np.load(str(path), allow_pickle=True) as data:
        loaded_dict = data['data'].item()
        return {str(k): v.astype(np.float32) for k, v in loaded_dict.items()}


def save_unaligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    path = unaligned_vectors_path(slice_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), data=np.array(embeddings, dtype=object))


def load_unaligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    path = unaligned_vectors_path(slice_id)
    if not path.exists():
        raise FileNotFoundError(f"Unaligned vectors missing: {path}")
    with np.load(str(path), allow_pickle=True) as data:
        loaded_dict = data["data"].item()
        return {str(k): v.astype(np.float32) for k, v in loaded_dict.items()}


# Embeddings:

def generate_embeddings_per_model(slice_range: tuple[int,int]) -> dict[str, np.ndarray]:
    """Train or load fastText model for slice."""
    model_file = slice_model_path(slice_range)
    if not model_file.exists():
        slice_file = SLICES_DIR / f"{slice_range[0]}-{slice_range[1]}.txt"
        if not slice_file.exists():
            raise FileNotFoundError(f"Training corpus missing for slice {slice_range}: {slice_file}")
        logger.info(f"Training fastText model for slice {slice_range} → {model_file}")
        model = fasttext.train_unsupervised(input=str(slice_file), **FASTTEXT_PARAMS)
        model.save_model(str(model_file))
    else:
        model = fasttext.load_model(str(model_file))
    return {str(tok): model.get_word_vector(tok).astype(np.float32) for tok in model.get_words()}


# Alignment:

def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common:
        return np.eye(len(next(iter(source_vectors.values()))), dtype=np.float32), source_vectors
    X = np.stack([source_vectors[w] for w in common])
    Y = np.stack([target_vectors[w] for w in common])
    R, _ = orthogonal_procrustes(X, Y)
    aligned = {w: vec @ R for w, vec in source_vectors.items()}
    return R, aligned


# FAISS:

def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    add_fn = cast(Callable[[np.ndarray], None], index.add)
    add_fn(vectors)


def build_index_for_slice(slice_range: tuple[int,int], use_aligned: bool = False) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned})")

    # Load embeddings from frozen files first
    if use_aligned:
        embeddings = load_aligned_vectors(slice_id)
    else:
        try:
            embeddings = load_unaligned_vectors(slice_id)
        except FileNotFoundError:
            embeddings = generate_embeddings_per_model(slice_range)
            save_unaligned_vectors(slice_id, embeddings)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings for slice {slice_id}, skipping")
        return

    vectors = np.stack([embeddings[w] for w in words])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    add_to_faiss_index(index, vectors)

    faiss.write_index(index, str(faiss_slice_path(slice_range, use_aligned)))
    with open(vocab_slice_path(slice_range, use_aligned), "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logger.info(f"Saved FAISS index and vocab for slice {slice_id}")


def search_index(
    index: faiss.Index,
    query: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    search_fn = cast(Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], index.search)
    return search_fn(query, k)


# Orchestration:

def build_all_slices(use_aligned: bool = False) -> None:
    """
    Build and persist vector spaces for all slices.

    Invariants:
    - Unaligned vectors are trained/loaded once and frozen to disk.
    - Aligned vectors are computed strictly from frozen unaligned vectors.
    - Diagnostics never retrains models unnecessarily.
    """

    if not use_aligned:
        # Stage 1: Freeze raw slice spaces
        for start, end in SLICES:
            slice_id = f"{start}-{end}"
            logger.info(f"Freezing unaligned vectors for {slice_id}")
            try:
                embeddings = load_unaligned_vectors(slice_id)
                logger.info(f"Loaded existing unaligned vectors for {slice_id}")
            except FileNotFoundError:
                embeddings = generate_embeddings_per_model((start, end))
                save_unaligned_vectors(slice_id, embeddings)
    else:
        # Stage 2: Align already-frozen spaces
        reference_slice_id = f"{SLICES[0][0]}-{SLICES[0][1]}"
        logger.info(f"Aligning all slices to reference {reference_slice_id}")

        ref_vectors = load_unaligned_vectors(reference_slice_id)
        save_aligned_vectors(reference_slice_id, ref_vectors)

        anchors_dict = get_anchors()
        ref_anchors = anchors_dict[reference_slice_id]["anchors"]

        for start, end in SLICES:
            slice_id = f"{start}-{end}"
            if slice_id == reference_slice_id:
                continue

            raw_vectors = load_unaligned_vectors(slice_id)
            _, aligned_vectors = orthogonal_procrustes_align(
                raw_vectors,
                ref_vectors,
                ref_anchors
            )
            save_aligned_vectors(slice_id, aligned_vectors)
            logger.info(f"Slice {slice_id} aligned")

    logger.info("Slice build complete.")



def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS indexes per slice")
    parser.add_argument("--aligned", action="store_true", help="Use aligned slice embeddings")
    args = parser.parse_args()
    env_aligned = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS")
    use_aligned = args.aligned or (env_aligned == "1")
    logger.info(f"Starting slice pipeline (aligned={use_aligned})")
    build_all_slices(use_aligned=use_aligned)


if __name__ == "__main__":
    main()
