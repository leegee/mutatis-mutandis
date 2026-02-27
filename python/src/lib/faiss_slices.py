#!/usr/bin/env python
"""
lib/faiss_slices.py

Utilities for managing FAISS slice indexes and vocabularies.
Provides loading, searching, and slice metadata for clients like audit/explorer.
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

USE_ALIGNED_FASTTEXT_VECTORS = True

EMB_DIM: int = int(config.FASTTEXT_PARAMS["dim"])


# Paths
def faiss_slice_path(slice_range: Tuple[int, int]) -> Path:
    start, end = slice_range
    return config.FAISS_INDEX_DIR / f"slice_{start}_{end}.faiss"


def vocab_slice_path(slice_range: Tuple[int, int]) -> Path:
    start, end = slice_range
    return config.FAISS_INDEX_DIR / f"slice_{start}_{end}.vocab"


#
# FastText
#

def load_fasttext_model(slice_range: Tuple[int, int]) -> Any:
    model_file = slice_model_path(slice_range)
    logger.info(f"Loading fastText model for slice {slice_range}: {model_file}")
    return fasttext.load_model(str(model_file))


def get_vector(conn, token: str, slice_start: int, slice_end: int) -> np.ndarray | None:
    """
    Fetch the canonical embedding from the token_vectors table.
    Returns L2-normalized vector or None if not found.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT vector
            FROM token_vectors
            WHERE token = %s
              AND slice_start = %s
              AND slice_end = %s
            LIMIT 1
            """,
            (token, slice_start, slice_end),
        )
        row = cur.fetchone()
    if row is None:
        return None
    vec = np.array(row[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec


# Load FAISS index + vocab
def load_slice_index(slice_range: Tuple[int, int]) -> tuple[Any, List[str]]:
    """Load FAISS index and corresponding vocabulary"""
    index: Any = faiss.read_index(str(faiss_slice_path(slice_range)))
    with open(vocab_slice_path(slice_range), "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f]
    return index, vocab


# KNN Search
def knn_search(index: Any, vocab: List[str], query_vec: np.ndarray, top_k: int = 25) -> List[Tuple[str, float]]:
    """
    Return top-k nearest tokens with cosine similarity.
    """
    query_vec = query_vec.reshape(1, -1).astype(np.float32)
    D, Idx = index.search(query_vec, top_k)
    results = [(vocab[idx], float(sim)) for idx, sim in zip(Idx[0], D[0], strict=True)]
    return results

