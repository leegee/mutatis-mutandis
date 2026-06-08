#!/usr/bin/env python
"""
lib/eebo_faiss.py

FAISS retrieval layer for EEBO semantic event embeddings.

This module intentionally treats FAISS as a *derived geometric index*,
not as a canonical data store.

Architectural role
------------------

    Postgres (identity + text provenance)
        |
    Zarr event log (canonical semantic events, sharded by corpus/period/strategy)
        |
    FAISS index (approximate geometric retrieval, one index per shard)

FAISS therefore owns ONLY:
    - vector geometry
    - event-id lookup
    - similarity search

It does NOT own:
    - metadata
    - provenance
    - semantic interpretation

Per-shard indexes
-----------------
Each Zarr shard (corpus / period / model / strategy) has its own FAISS
index at the mirrored path under FAISS_ROOT.  The search layer composes
results across shards as needed.  faiss_path_for_shard() in
tier1_5_build_faiss_index.py handles path resolution.

Vector reconstruction
---------------------
EeboFaissIndex exposes reconstruct() for IndexFlatIP indexes, which store
vectors verbatim.  This is NOT supported by IndexHNSWFlat.  See method
docstring for migration notes.

Core invariant
--------------
Every indexed vector corresponds to a stable observation event_id.
All vectors are unit-normalised before insertion so that:

    inner product == cosine similarity

This guarantees stable geometric interpretation across ingestion,
querying, and persistence.
"""

from __future__ import annotations

import shutil
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
        - cosine similarity semantics (unit-normalised inner product)
        - persistence validation
        - future-compatible with HNSW migration
    """

    def __init__(self, dim: int, exact: bool = True):
        self.dim = dim

        if exact:
            self.base = faiss.IndexFlatIP(dim)
        else:
            # M=32 is a good general-purpose HNSW default
            self.base = faiss.IndexHNSWFlat(dim, 32)
            self.base.metric_type = faiss.METRIC_INNER_PRODUCT

        self._index = faiss.IndexIDMap(self.base)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[faiss] saving index={path} ntotal={self._index.ntotal}")
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "EeboFaissIndex":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"FAISS index not found: {path}")

        logger.info(f"[faiss] loading index={path}")

        obj        = cls.__new__(cls)
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
            raise TypeError(
                f"Cannot infer embedding dimension from FAISS index "
                f"of type {type(base).__name__}. Expected an index with "
                f"a '.d' attribute (e.g. IndexFlatIP, IndexHNSWFlat)."
            )

        logger.info(
            f"[faiss] loaded ntotal={obj._index.ntotal} dim={obj.dim}"
        )
        return obj

    @staticmethod
    def wipe_faiss_index(path: Path) -> None:
        """
        Delete a persisted FAISS index from disk.
        Must always be followed by a rebuild.
        """
        path = Path(path)
        logger.info(f"[faiss] deleting index={path}")

        if path.is_file():
            path.unlink()
            logger.info(f"[faiss] deleted file index={path}")
        elif path.exists():
            shutil.rmtree(path)
            logger.info(f"[faiss] deleted directory index={path}")
        else:
            logger.info(f"[faiss] no index found at={path}")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, event_ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids     = np.ascontiguousarray(event_ids, dtype=np.int64)

        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n, dim)")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim mismatch: expected {self.dim}, "
                f"got {vectors.shape[1]}"
            )

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError(
                "number of vectors must match number of event IDs"
            )

        if np.any(ids == -1):
            raise ValueError("Invalid FAISS ids (-1) detected")

        # Guard against within-batch duplicates
        seen: set[int] = set()
        for eid in ids:
            eid = int(eid)
            if eid in seen:
                raise ValueError(f"Duplicate event_id in batch: {eid}")
            seen.add(eid)

        # Guard against cross-call duplicates within this index
        if self._index.ntotal > 0:
            existing    = set(faiss.vector_to_array(self._index.id_map).tolist())
            cross_dupes = [int(eid) for eid in ids if int(eid) in existing]
            if cross_dupes:
                raise ValueError(
                    f"event_ids already present in index: "
                    f"{cross_dupes[:10]}"
                    f"{'...' if len(cross_dupes) > 10 else ''}"
                )

        vectors = self._normalize(vectors, event_ids=ids)
        self._index.add_with_ids(vectors, ids)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        queries: np.ndarray,
        k:       int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest semantic neighbours.

        Returns
        -------
        similarities : (n_queries, k) float32 — cosine similarity scores
        event_ids    : (n_queries, k) int64   — observation identifiers
        """
        queries = np.asarray(queries, dtype=np.float32)

        if queries.ndim == 1:
            queries = queries[None, :]

        if queries.shape[1] != self.dim:
            raise ValueError(
                f"query dim mismatch: expected {self.dim}, "
                f"got {queries.shape[1]}"
            )

        queries = self._normalize(queries)
        scores, ids = self._index.search(queries, k)
        return scores, ids

    def reconstruct(self, event_id: int) -> np.ndarray:
        """
        Retrieve the unit-normalised vector stored for event_id.

        Only supported for IndexFlatIP (which stores vectors verbatim).
        Raises RuntimeError for IndexHNSWFlat.

        Returns
        -------
        (dim,) float32 — the L2-normalised vector as inserted.
        """
        base = self._index.index

        if not isinstance(base, faiss.IndexFlatIP):
            raise RuntimeError(
                f"reconstruct() is only supported for IndexFlatIP. "
                f"Current base index is {type(base).__name__}, which does "
                f"not support vector reconstruction.  Query vectors must be "
                f"sourced from Zarr instead."
            )

        vec = np.zeros(self.dim, dtype=np.float32)
        self._index.reconstruct(int(event_id), vec)
        return vec

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(
        x:         np.ndarray,
        event_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        x     = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        zero_mask = (norms == 0).ravel()

        if np.any(zero_mask):
            zero_positions = np.where(zero_mask)[0].tolist()
            if event_ids is not None:
                offending = [
                    int(np.asarray(event_ids)[i]) for i in zero_positions
                ]
                raise ValueError(
                    f"Zero vector at batch positions {zero_positions}, "
                    f"event_ids={offending}. "
                    f"Indicates invalid embedding generation upstream."
                )
            else:
                raise ValueError(
                    f"Zero vector at batch positions {zero_positions}. "
                    f"Indicates invalid embedding generation upstream."
                )

        return x / norms
