#!/usr/bin/env python
"""
build_faiss_slice_indexes.py

Build a FAISS index for every temporal slice using its fastText model.
Each index is saved alongside a vocabulary file so indices → tokens can be resolved.
"""

from __future__ import annotations
from typing import Any, List, Tuple

from pathlib import Path
import numpy as np
import faiss
import fasttext

import lib.eebo_config as config
from lib.eebo_logging import logger
from generate_token_embeddings import slice_model_path

EMB_DIM: int = int(config.FASTTEXT_PARAMS["dim"])

def faiss_slice_path(slice_range: Tuple[int, int]) -> Path:
    """
    Return the faiss index path for a given slice.
    slice_range: (start_year, end_year)
    """
    start, end = slice_range
    return config.FAISS_INDEX_DIR / f"slice_{start}_{end}.faiss"


def vocab_slice_path(slice_range: Tuple[int, int]) -> Path:
    """
    Return the faiss vocab path for a given slice.
    slice_range: (start_year, end_year)
    """
    start, end = slice_range
    return config.FAISS_INDEX_DIR / f"slice_{start}_{end}.vocab"

def load_fasttext_model(slice_range: Tuple[int, int]) -> Any:
    """Load a fastText slice model"""
    model_file = slice_model_path(slice_range)
    logger.info(f"Loading fastText model: {model_file}")
    return fasttext.load_model(str(model_file))


def build_index_for_slice(slice_range: Tuple[int, int]) -> None:
    start, end = slice_range
    model = load_fasttext_model(slice_range)

    logger.info(f"Extracting vocabulary for slice {start}-{end}")
    words: List[str] = model.get_words()

    # Build embedding matrix with explicit typing
    vectors = np.empty((len(words), EMB_DIM), dtype=np.float32)

    for i, w in enumerate(words):
        vec = model.get_word_vector(w)
        vectors[i] = np.asarray(vec, dtype=np.float32)

    # L2-normalise for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    vectors = vectors / norms

    logger.info(f"Building FAISS index ({len(words)} vectors)")
    index: Any = faiss.IndexFlatIP(EMB_DIM)  # IP + normalized vectors = cosine
    index.add(vectors)

    # Save index
    # index_path = config.FAISS_INDEX_DIR / f"slice_{start}_{end}.faiss"
    index_path = faiss_slice_path((start, end))
    faiss.write_index(index, str(index_path))

    # Save vocab mapping
    vocab_path = config.FAISS_INDEX_DIR / f"slice_{start}_{end}.vocab"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))

    logger.info(f"Saved FAISS index and vocab for slice {start}-{end}")


def main() -> None:
    logger.info("Building FAISS slice indexes")

    for slice_range in config.SLICES:
        build_index_for_slice(slice_range)

    logger.info("All FAISS indexes built")


if __name__ == "__main__":
    main()
