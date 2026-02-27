#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (aligned or unaligned) and build FAISS indexes.

- Embeddings: train or load fastText models per slice, or use pre-aligned NumPy arrays
- FAISS indices: stored per slice in separate subdirs for aligned vs unaligned
- Respects env var USE_ALIGNED_FASTTEXT_VECTORS and CLI flag --aligned
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from lib.eebo_logging import logger
from typing import Callable, cast
import fasttext
import numpy as np
import faiss

from lib.eebo_config import (
    FASTTEXT_SLICE_MODEL_DIR, SLICES, SLICES_DIR,
    FAISS_INDEX_DIR, ALIGNED_VECTORS_DIR, FASTTEXT_PARAMS
)


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


# Embeddings

def generate_embeddings_per_model(slice_range: tuple[int,int]) -> dict[str, np.ndarray]:
    """Load or train a fastText slice model and return embeddings keyed by str -> ndarray[float32]."""
    model_file = slice_model_path(slice_range)
    if not model_file.exists():
        # Train model if missing
        slice_file = SLICES_DIR / f"{slice_range[0]}-{slice_range[1]}.txt"
        if not slice_file.exists():
            raise FileNotFoundError(f"Training corpus missing for slice {slice_range}: {slice_file}")
        logger.info(f"Training fastText model for slice {slice_range} → {model_file}")
        model = fasttext.train_unsupervised(input=str(slice_file), **FASTTEXT_PARAMS)
        model.save_model(str(model_file))
    else:
        model = fasttext.load_model(str(model_file))

    return {str(tok): model.get_word_vector(tok).astype(np.float32) for tok in model.get_words()}


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    """Load pre-aligned slice embeddings from ALIGNED_VECTORS_DIR."""
    path = ALIGNED_VECTORS_DIR / f"{slice_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Aligned vectors missing: {path}")
    data = np.load(path, allow_pickle=True).item()
    return {str(k): v.astype(np.float32) for k, v in data.items()}


def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
    """
    Safely add vectors to a FAISS index.

    - Ensures float32 dtype
    - Ensures contiguous memory layout
    - Works around incorrect FAISS type stubs (add(self, n, x))
    - Keeps Pyright happy without type: ignore
    """
    # Ensure correct memory layout + dtype
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    # Work around incorrect FAISS stub signature
    add_fn = cast(Callable[[np.ndarray], None], index.add)
    add_fn(vectors)


def build_index_for_slice(slice_range: tuple[int,int], use_aligned: bool = False) -> None:
    start, end = slice_range
    slice_id = f"{start}-{end}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned})")

    # Load embeddings
    if use_aligned:
        embeddings = load_aligned_vectors(slice_id)
    else:
        embeddings = generate_embeddings_per_model(slice_range)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings found for slice {slice_id}, skipping")
        return

    vectors: np.ndarray = np.stack([embeddings[w] for w in words], axis=0)

    # L2 normalize (before FAISS)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    dim: int = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)

    # Safe FAISS call
    add_to_faiss_index(index, vectors)

    # Save index + vocab
    faiss.write_index(index, str(faiss_slice_path(slice_range, use_aligned)))
    with open(vocab_slice_path(slice_range, use_aligned), "w", encoding="utf-8") as f:
        f.write("\n".join(words))

    logger.info(f"Saved FAISS index and vocab for slice {slice_id}")


def build_all_slices(use_aligned: bool = False) -> None:
    for slice_range in SLICES:
        build_index_for_slice(slice_range, use_aligned)


# CLI entrypoint

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS indexes per slice")
    parser.add_argument(
        "--aligned", action="store_true",
        help="Use aligned slice embeddings instead of training fastText"
    )
    args = parser.parse_args()

    # Respect env var if set
    env_aligned = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS")
    use_aligned = args.aligned or (env_aligned == "1")

    logger.info(f"Starting slice pipeline (aligned={use_aligned})")
    build_all_slices(use_aligned=use_aligned)


if __name__ == "__main__":
    main()
