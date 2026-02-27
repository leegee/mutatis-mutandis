#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (aligned or unaligned) and build FAISS indexes.

- Embeddings: train or load fastText models per slice, or compute aligned embeddings
- FAISS indices: stored per slice in separate subdirs for aligned vs unaligned
- Respects env var USE_ALIGNED_FASTTEXT_VECTORS and CLI flag --aligned
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


# Paths

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
    """Save aligned embeddings safely for Windows and Mypy."""
    ALIGNED_VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    path_str = str(aligned_vectors_path(slice_id))
    # Wrap dict in a 0-d object array
    np.savez(path_str, data=np.array(embeddings, dtype=object))


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    """Load aligned embeddings safely."""
    path = aligned_vectors_path(slice_id)
    if not path.exists():
        raise FileNotFoundError(f"Aligned vectors missing: {path}")

    with np.load(str(path), allow_pickle=True) as data:
        # Unpack object array
        loaded_dict = data['data'].item()
        return {str(k): v.astype(np.float32) for k, v in loaded_dict.items()}


# Embeddings

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


# Alignment

def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common:
        return np.eye(len(next(iter(source_vectors.values()))), dtype=np.float32), source_vectors
    X = np.stack([source_vectors[w] for w in common])
    Y = np.stack([target_vectors[w] for w in common])
    R, _ = orthogonal_procrustes(X, Y)
    aligned = {w: vec @ R for w, vec in source_vectors.items()}
    return R, aligned


def compute_aligned_slices(reference_slice_id: str) -> dict[str, dict[str, np.ndarray]]:
    """Train/load slices and align all to reference slice.

    Saves only **non-reference slices** as they are aligned.
    """
    anchors_dict = get_anchors()
    slice_ids = [f"{start}-{end}" for start, end in SLICES]
    aligned_embeddings: dict[str, dict[str, np.ndarray]] = {}

    # Load/generate reference slice
    start, end = map(int, reference_slice_id.split("-"))
    ref_vectors = generate_embeddings_per_model((start, end))
    ref_anchors = anchors_dict[reference_slice_id]["anchors"]
    aligned_embeddings[reference_slice_id] = ref_vectors

    for sid in slice_ids:
        if sid == reference_slice_id:
            continue
        start, end = map(int, sid.split("-"))
        vectors = generate_embeddings_per_model((start, end))
        _, aligned_vectors = orthogonal_procrustes_align(vectors, ref_vectors, ref_anchors)
        aligned_embeddings[sid] = aligned_vectors
        save_aligned_vectors(sid, aligned_vectors)
        logger.info(f"Slice {sid} aligned to reference slice {reference_slice_id}")

    # Save reference slice aligned vectors **once**
    save_aligned_vectors(reference_slice_id, ref_vectors)
    logger.info("All slices aligned successfully")
    return aligned_embeddings


# FAISS

def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    add_fn = cast(Callable[[np.ndarray], None], index.add)
    add_fn(vectors)


def build_index_for_slice(slice_range: tuple[int,int], use_aligned: bool = False) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned})")

    embeddings = load_aligned_vectors(slice_id) if use_aligned else generate_embeddings_per_model(slice_range)

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


# Orchestration

def build_all_slices(use_aligned: bool = False) -> None:
    if use_aligned:
        # Align all slices to first slice
        reference_slice_id = f"{SLICES[0][0]}-{SLICES[0][1]}"
        compute_aligned_slices(reference_slice_id)
        for slice_range in SLICES:
            build_index_for_slice(slice_range, use_aligned=True)
    else:
        for slice_range in SLICES:
            build_index_for_slice(slice_range, use_aligned=False)


# CLI

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
