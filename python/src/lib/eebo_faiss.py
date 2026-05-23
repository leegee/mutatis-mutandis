#!/usr/bin/env python
"""
eebo_faiss.py

FAISS retrieval layer for EEBO semantic event embeddings.

This module intentionally treats FAISS as a *derived geometric index*,
not as a canonical data store.

Architectural role
------------------

The EEBO embedding pipeline now operates as:

    Postgres (identity + text provenance)
        ↓
    Zarr event log (canonical semantic events)
        ↓
    FAISS index (approximate geometric retrieval)

FAISS therefore owns ONLY:
    - vector geometry
    - event-id lookup
    - similarity search

It does NOT own:
    - metadata
    - provenance
    - semantic interpretation
    - vector reconstruction

Core invariant
--------------

Every indexed vector corresponds to a stable semantic event ID.

An event ID should uniquely identify:
    - a token occurrence
    - in a document
    - at a specific token position

All vectors are unit-normalised before insertion so that:

    inner product == cosine similarity

This guarantees stable geometric interpretation across:
    - ingestion
    - querying
    - persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import faiss
import numpy as np

from lib.eebo_logging import logger


class EeboFaissIndex:
    """
    Thin wrapper around FAISS IndexIDMap.

    Design goals:
        - explicit semantic event IDs
        - cosine similarity semantics
        - persistence validation
        - future-compatible with HNSW migration
    """

    def __init__(self, dim: int, exact: bool = True):
        self.dim = dim

        if exact:
            base = faiss.IndexFlatIP(dim)
        else:
            # future-friendly approximate mode
            # M=32 is a good general-purpose HNSW default
            base = faiss.IndexHNSWFlat(dim, 32)
            base.metric_type = faiss.METRIC_INNER_PRODUCT

        self._index = faiss.IndexIDMap(base)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """
        Enforce cosine/IP equivalence.

        Failure mode:
            zero vectors imply invalid embedding generation upstream.
        """

        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Zero vector encountered during normalization")
        return x / norms

    def add(self, vectors: np.ndarray, event_ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(event_ids, dtype=np.int64)

        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n, dim)")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                "number of vectors must match number of event IDs"
            )

        if np.any(ids == -1):
            raise ValueError("Invalid FAISS ids (-1) detected")

        seen = set()
        for eid in ids:
            eid = int(eid)
            if eid in seen:
                raise ValueError(f"Duplicate event_id in batch: {eid}")
            seen.add(eid)

        vectors = self._normalize(vectors)
        self._index.add_with_ids(vectors, ids)

    def search(
        self,
        queries: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest semantic neighbours.

        Returns:
            similarities, event_ids

        similarities:
            cosine similarity scores in descending order

        event_ids:
            semantic event identifiers corresponding to neighbours
        """

        queries = np.asarray(queries, dtype=np.float32)

        if queries.ndim == 1:
            queries = queries[None, :]

        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim mismatch: expected {self.dim}, got {queries.shape[1]}"
            )

        queries = self._normalize(queries)

        scores, ids = self._index.search(queries, k)

        return scores, ids

    @property
    def ntotal(self) -> int:
        """
        Number of indexed semantic events.
        """
        return self._index.ntotal

    def save(self, path: Path) -> None:
        """
        Persist FAISS index.

        Persistence invariant:
            metric geometry must survive round-trip.
        """

        path = Path(path)

        logger.info(f"[faiss] saving index={path}")

        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "EeboFaissIndex":
        """
        Load persisted FAISS index and validate geometry invariants.
        """

        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        logger.info(f"[faiss] loading index={path}")

        obj = cls.__new__(cls)

        obj._index = faiss.read_index(str(path))

        if not isinstance(obj._index, faiss.IndexIDMap):
            raise TypeError(
                "Loaded FAISS index must be IndexIDMap "
                "(semantic IDs are required)"
            )

        base = obj._index.index

        if not hasattr(base, "metric_type"):
            raise TypeError(
                f"Cannot determine metric type for index: {type(base)}"
            )

        if base.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise TypeError(
                "FAISS index must use INNER_PRODUCT "
                "(cosine similarity invariant)"
            )

        if hasattr(base, "d"):
            obj.dim = base.d
        else:
            raise TypeError("Cannot infer dimension from FAISS index")

        logger.info(
            f"[faiss] loaded ntotal={obj._index.ntotal} dim={obj.dim}"
        )

        return obj
