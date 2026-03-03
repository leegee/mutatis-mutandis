#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (aligned or unaligned) and build FAISS indexes.

- Embeddings: train or load fastText models per slice, or compute aligned embeddings
- FAISS indices: stored per slice in separate subdirs for aligned vs unaligned
- Provides methods to add and search the index
- Env var `USE_ALIGNED_FASTTEXT_VECTORS` and flag `--aligned`

    Corpus (slice text file: SLICES_DIR/1625-1629.txt)
            │
            V
    [Train fastText model once]
            │
            V
    Slice fastText model (slice_1625_1629.bin)
    * Stored in FASTTEXT_SLICE_MODEL_DIR
    * Trained once per slice
    * Can be reused for:
        - Unaligned embeddings
        - Aligned embeddings
            │
            V
    Generate Unaligned Vectors
    (raw embeddings from fastText)
            │
            V
    Unaligned vectors (slice_1625_1629.npz)
    * Stored in UNALIGNED_VECTORS_DIR
    * Used to build:
        - FAISS index (unaligned)
            │
            V
    FAISS index (slice_1625_1629.faiss, unaligned)
    * Stored in UNALIGNED_VECTORS_DIR
    * Queries raw vectors
    * Vocabulary mapping stored in slice_1625_1629.vocab
            │
            V
    Align Vectors to Reference Slice
    * Takes unaligned vectors
    * Applies orthogonal Procrustes rotation
    * Preserves terms (vocab), only vector coordinates change
            │
            V
    Aligned vectors (slice_1625_1629.npz)
    * Stored in ALIGNED_VECTORS_DIR
            │
            V
    FAISS index (slice_1625_1629.faiss, aligned)
    * Stored in ALIGNED_VECTORS_DIR
    * Queries aligned vectors
    * Vocabulary mapping shared with unaligned (terms unchanged)

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
    UNALIGNED_VECTORS_DIR, ALIGNED_VECTORS_DIR, FASTTEXT_PARAMS
)
from lib.eebo_anchor_builder import get_anchors


# Paths

def unaligned_vectors_path(slice_id: str) -> Path:
    return UNALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def aligned_vectors_path(slice_id: str) -> Path:
    return ALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    return FASTTEXT_SLICE_MODEL_DIR / f"slice_{start}_{end}.bin"


def faiss_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = (ALIGNED_VECTORS_DIR if aligned else UNALIGNED_VECTORS_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.faiss"


def vocab_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = (ALIGNED_VECTORS_DIR if aligned else UNALIGNED_VECTORS_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.vocab"



# Internal IO

def _save_vectors(path: Path, embeddings: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Prefix keys to avoid NumPy argument collision
    safe_dict = {f"tok_{k}": v for k, v in embeddings.items()}
    np.savez_compressed(path, **safe_dict)


def _load_vectors(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Vectors missing: {path}")
    with np.load(path) as data:
        return {
            k.removeprefix("tok_"): data[k].astype(np.float32)
            for k in data
        }

# Public IO

def save_aligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    _save_vectors(aligned_vectors_path(slice_id), embeddings)


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    return _load_vectors(aligned_vectors_path(slice_id))


def save_unaligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    _save_vectors(unaligned_vectors_path(slice_id), embeddings)


def load_unaligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    return _load_vectors(unaligned_vectors_path(slice_id))


# Embeddings

def generate_embeddings_per_model(slice_range: tuple[int,int]) -> dict[str, np.ndarray]:
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

# Probably easier to outsource this
def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common:
        logger.warning("No anchors found; skipping alignment for this slice")
        return np.eye(len(next(iter(source_vectors.values()))), dtype=np.float32), source_vectors
    X = np.stack([source_vectors[w] for w in common])
    Y = np.stack([target_vectors[w] for w in common])
    R, _ = orthogonal_procrustes(X, Y)
    aligned = {w: vec @ R for w, vec in source_vectors.items()}
    # Normalize aligned vectors
    aligned = {k: v / np.linalg.norm(v) for k, v in aligned.items()}
    return R, aligned


# Public FAISS

def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    # cafeful of mypy
    add_fn = cast(Callable[[np.ndarray], None], index.add)
    add_fn(vectors)


def build_index_for_slice(slice_range: tuple[int,int], use_aligned: bool = False) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned})")

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


def search_index(index: faiss.Index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    search_fn = cast(Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], index.search)
    return search_fn(query, k)


# Orchestration

def build_all_slices(use_aligned: bool = False) -> None:
    logger.info("Building slices (aligned=%s)", use_aligned)

    # Stage 1: produce & save unaligned
    unaligned_cache: dict[str, dict[str, np.ndarray]] = {}

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        span = (start, end)

        try:
            vectors = load_unaligned_vectors(slice_id)
            logger.info("Loaded unaligned vectors for %s", slice_id)
        except FileNotFoundError:
            vectors = generate_embeddings_per_model(span)
            save_unaligned_vectors(slice_id, vectors)
            logger.info("Generated unaligned vectors for %s", slice_id)

        unaligned_cache[slice_id] = vectors
        build_index_for_slice(span, use_aligned=False)

    if not use_aligned:
        logger.info("Unaligned build complete.")
        return

    # Stage 2: align slices to center slice
    mid_index = len(SLICES) // 2
    reference_span = SLICES[mid_index]
    reference_id = f"{reference_span[0]}-{reference_span[1]}"
    ref_vectors = unaligned_cache[reference_id]

    anchors_dict = get_anchors()
    ref_anchors = anchors_dict[reference_id]["anchors"]

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        span = (start, end)

        if slice_id == reference_id:
            aligned_vectors = ref_vectors
        else:
            raw_vectors = unaligned_cache[slice_id]
            _, aligned_vectors = orthogonal_procrustes_align(
                raw_vectors,
                ref_vectors,
                ref_anchors,
            )

        save_aligned_vectors(slice_id, aligned_vectors)
        build_index_for_slice(span, use_aligned=True)
        logger.info("Aligned and indexed %s", slice_id)

    logger.info("Aligned build complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS indexes per slice")
    parser.add_argument("--aligned", action="store_true", help="Use aligned slice embeddings")
    args = parser.parse_args()
    env_aligned = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS", "").lower()
    use_aligned = args.aligned or env_aligned in {"1", "true", "yes"}
    logger.info(f"Starting slice pipeline (aligned={use_aligned})")
    build_all_slices(use_aligned=use_aligned)


if __name__ == "__main__":
    main()
