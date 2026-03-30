#!/usr/bin/env python3
"""
lib/faiss_search_vectors.py

FAISS search and bulk vector retrieval for occurrence-level embeddings.
"""

import numpy as np
from typing import Sequence, Tuple
from lib.FaissIndex import FaissIndex


class SliceVectors:
    """
    Load occurrence-level vectors for a slice and allow fast bulk lookup by token_occurrence_id.
    """
    def __init__(self, vectors_path: str):
        data = np.load(vectors_path, allow_pickle=True)
        self.vectors: np.ndarray = data["vectors"].astype(np.float32)
        self.ids: np.ndarray = data["token_occurrence_id"] if "token_occurrence_id" in data else data["doc_ids"]
        self.id_to_pos: dict[int, int] = {id_: i for i, id_ in enumerate(self.ids)}

    def get_vectors_by_ids(self, query_ids: Sequence[int]) -> np.ndarray:
        """
        Return a 2D array (n_query, dim) of vectors corresponding to query_ids
        """
        pos = [self.id_to_pos[qid] for qid in query_ids]
        return self.vectors[pos]


def faiss_search_with_vectors(
    index: FaissIndex,
    slice_vectors: SliceVectors,
    query_vec: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS for nearest occurrences and return:
      - token_occurrence_ids
      - corresponding vectors from the slice

    Args:
        index: OccurrenceFaissIndex
        slice_vectors: SliceVectors object
        query_vec: (1, dim) normalized query vector
        k: number of neighbors

    Returns:
        tuple: (ids: np.ndarray shape (k,), vectors: np.ndarray shape (k, dim))
    """
    distances, ids = index.search(query_vec, k)
    ids = ids[0]  # single query
    valid_ids = ids[ids != -1]
    vectors = slice_vectors.get_vectors_by_ids(valid_ids)
    return valid_ids, vectors
