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

Vector reconstruction
---------------------

EeboFaissIndex exposes a reconstruct() method that retrieves the
unit-normalised vector stored for a given event_id directly from the
FAISS index, without reaching back into the Zarr store.

This is currently supported because the pipeline uses IndexFlatIP, which
stores vectors verbatim. It is NOT supported by IndexHNSWFlat, which does
not retain vectors after index construction.

The intended future use is in ZarrEventLookup (tier2_concept_neighbours.py):
once the index type is confirmed stable, the "embedding" field can be
dropped from by_event_id and replaced with reconstruct() calls, eliminating
the in-memory copy of the full corpus embedding matrix. Until that migration
is made, Zarr remains the authoritative source for query vectors and
reconstruct() is provided but not called.

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
        - cosine similarity semantics
        - persistence validation
        - future-compatible with HNSW migration
    """

    def __init__(self, dim: int, exact: bool = True):
        self.dim = dim

        if exact:
            self.base = faiss.IndexFlatIP(dim)
        else:
            # future-friendly approximate mode
            # M=32 is a good general-purpose HNSW default
            self.base = faiss.IndexHNSWFlat(dim, 32)
            self.base.metric_type = faiss.METRIC_INNER_PRODUCT

        self._index = faiss.IndexIDMap(self.base)
        self._ids = set()

    @staticmethod
    def wipe_faiss_index(path: Path) -> None:
        """
        Deletes persisted FAISS index from disk.

        Failure mode:
            - if file is in use, OS will raise
            - if path is wrong, silent mismatch risk upstream

        This must always be followed by rebuild_from_tier1().
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

        logger.info(f"[faiss] deleted index={path}")

    @staticmethod
    def _normalize(
        x: np.ndarray,
        event_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Enforce cosine/IP equivalence.

        Failure mode:
            zero vectors imply invalid embedding generation upstream.
            event_ids, if provided, are included in the error to identify
            the offending observations.
        """

        x = np.asarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        zero_mask = (norms == 0).ravel()

        if np.any(zero_mask):
            zero_positions = np.where(zero_mask)[0].tolist()
            if event_ids is not None:
                offending = [
                    int(np.asarray(event_ids)[i]) for i in zero_positions
                ]
                raise ValueError(
                    f"Zero vector encountered during normalisation at batch "
                    f"positions {zero_positions}, event_ids={offending}. "
                    f"This indicates invalid embedding generation upstream."
                )
            else:
                raise ValueError(
                    f"Zero vector encountered during normalisation at batch "
                    f"positions {zero_positions}. "
                    f"This indicates invalid embedding generation upstream."
                )
        return x / norms

    def add(self, vectors: np.ndarray, event_ids: Sequence[int]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        ids     = np.ascontiguousarray(event_ids, dtype=np.int64)

        if vectors.ndim != 2:
            raise ValueError("vectors must have shape (n, dim)")

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim mismatch: expected {self.dim}, got {vectors.shape[1]}"
            )

        if vectors.shape[0] != ids.shape[0]:
            raise ValueError("number of vectors must match number of event IDs")

        if np.any(ids == -1):
            raise ValueError("Invalid FAISS ids (-1) detected")

        # Guard against within-batch duplicates
        seen = set()
        for eid in ids:
            eid = int(eid)
            if eid in seen:
                raise ValueError(f"Duplicate event_id in batch: {eid}")
            seen.add(eid)

        # Guard against cross-call duplicates
        if self._index.ntotal > 0:
            existing    = set(faiss.vector_to_array(self._index.id_map).tolist())
            cross_dupes = [int(eid) for eid in ids if int(eid) in existing]
            if cross_dupes:
                raise ValueError(
                    f"event_ids already present in index: {cross_dupes[:10]}"
                    f"{'...' if len(cross_dupes) > 10 else ''}"
                )

        vectors = self._normalize(vectors, event_ids=ids)
        self._index.add_with_ids(vectors, ids)
        self._ids.update(int(i) for i in ids)

    def ids(self) -> set[int]:
        return self._ids

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

    def reconstruct(self, event_id: int) -> np.ndarray:
        """
        Retrieve the unit-normalised vector stored for event_id.

        Returns:
            (dim,) float32 array — the L2-normalised vector as it was
            inserted, i.e. suitable for direct inner-product comparison.

        Index-type constraint:
            This method is only supported when the underlying base index is
            IndexFlatIP, which stores vectors verbatim. It will raise a
            RuntimeError for IndexHNSWFlat, which discards vectors after
            construction and does not support reconstruction.

            Before calling this method, callers should confirm the index
            type via isinstance(self._index.index, faiss.IndexFlatIP).

        Intended use (deferred migration):
            ZarrEventLookup in tier2_concept_neighbours.py currently stores
            a copy of every embedding in its by_event_id dict, holding the
            full corpus embedding matrix in memory. Once the index type is
            confirmed stable as IndexFlatIP, that "embedding" field can be
            dropped and replaced with calls to this method, eliminating the
            duplicate copy. The migration is deferred because switching to
            IndexHNSWFlat would silently break any caller relying on
            reconstruct().

        Note:
            Reconstructed vectors are the normalised form stored in FAISS,
            not the raw pre-normalisation vectors from Zarr. For cosine
            similarity purposes these are equivalent, but for any use case
            that requires the original unnormalised embedding, Zarr remains
            the authoritative source.
        """

        base = self._index.index

        if not isinstance(base, faiss.IndexFlatIP):
            raise RuntimeError(
                f"reconstruct() is only supported for IndexFlatIP. "
                f"Current base index is {type(base).__name__}, which does "
                f"not support vector reconstruction. Query vectors must be "
                f"sourced from Zarr instead."
            )

        vec = np.zeros(self.dim, dtype=np.float32)
        self._index.reconstruct(int(event_id), vec)
        return vec

    def save(self, path: Path) -> None:
        """
        Persist FAISS index.

        Persistence invariant:
            metric geometry must survive round-trip.
        """
        path = Path(path)
        logger.info(f"[faiss] saving index={path} ntotal={self._index.ntotal}")
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

        obj = cls(dim=1)  # placeholder
        obj._index = faiss.read_index(str(path))

        if not isinstance(obj._index, faiss.IndexIDMap):
            raise TypeError(
                "Loaded FAISS index must be IndexIDMap "
                "(semantic IDs are required)"
            )

        base = obj._index.index
        obj.dim = base.d # replace placeholder

        if not hasattr(base, "metric_type"):
            raise TypeError(
                f"Cannot determine metric type for index: {type(base)}"
            )

        if base.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise TypeError(
                "FAISS index must use INNER_PRODUCT "
                "(cosine similarity invariant)"
            )

        # Both IndexFlatIP and IndexHNSWFlat expose .d for the vector
        # dimension. If this ever fails for a future index type, the
        # TypeError below will surface it explicitly rather than
        # letting obj.dim remain unset.
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
