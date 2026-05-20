"""
lib.zarr_embedding_eventlog.py

ZarrEmbeddingEventLog - canonical EEBO event store.

This module defines a single-slice, append-only event log for
contextualised embedding inference.

Each row is a semantic event:

    event = (
        vector_id,   # stable lexical identity (Postgres-backed)
        doc_id,      # document provenance
        token_idx,   # token position within document
        emb_norm,    # normalised contextual embedding (for cosine space)
        emb_raw      # raw contextual embedding (model space)
    )

Design invariants
------------------

1. Event atomicity
   Each row is an independent semantic observation.

2. Column alignment is structural, not logical
   Arrays are guaranteed aligned only by append contract.

3. Append-only semantics
   No mutation, deletion, or rewriting of existing events.

4. Single-store responsibility
   This object represents ONE Zarr event store (typically one slice).

Cross-slice composition is handled externally (streaming layer).

Failure modes
-------------
- Concurrent writers will break determinism
- Partial writes are not transactional
- Zero-length embeddings are rejected implicitly upstream
"""

from __future__ import annotations

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrEmbeddingEventLog:
    """
    Append-only event log over Zarr arrays.

    Intended usage:
        - ingestion pipeline (tier1_corpus2zarr.py)
        - slice-level analysis/debugging
    """

    def __init__(self, path: str, dim: int):
        self.root = zarr.open_group(path, mode="a", zarr_version=2)
        self.dim = dim

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        g = self.root.require_group("events")

        self.emb_norm = self._ds(g, "emb_norm", (dim,), compressor, "float32")
        self.emb_raw = self._ds(g, "emb_raw", (dim,), compressor, "float32")

        self.vector_id = self._ds(g, "vector_id", (), compressor, "int64")
        self.token_idx = self._ds(g, "token_idx", (), compressor, "int64")
        self.doc_id = self._ds(g, "doc_id", (), compressor, "U32")

    # ------------------------------------------------------------
    # dataset helpers
    # ------------------------------------------------------------

    def _ds(self, g, name, shape_suffix, compressor, dtype):
        if name in g:
            return g[name]

        shape = (0,) + shape_suffix

        chunks = (4096,) if len(shape_suffix) == 0 else (4096, shape_suffix[0])

        return g.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )

    # ------------------------------------------------------------
    # core write path
    # ------------------------------------------------------------

    def append_events(
        self,
        emb_norm,
        emb_raw,
        vector_id,
        doc_id,
        token_idx,
    ):
        emb_norm = np.asarray(emb_norm, dtype=np.float32)
        emb_raw = np.asarray(emb_raw, dtype=np.float32)
        vector_id = np.asarray(vector_id, dtype=np.int64)
        token_idx = np.asarray(token_idx, dtype=np.int64)
        doc_id = np.asarray(doc_id, dtype=object)

        n = vector_id.shape[0]

        self._check(emb_norm, n)
        self._check(emb_raw, n)
        self._check(vector_id, n)
        self._check(token_idx, n)
        self._check(doc_id, n)

        self._append(self.emb_norm, emb_norm)
        self._append(self.emb_raw, emb_raw)
        self._append(self.vector_id, vector_id)
        self._append(self.token_idx, token_idx)
        self._append(self.doc_id, doc_id)

    def _check(self, arr, n):
        if len(arr) != n:
            raise ValueError(f"event size mismatch: expected {n}, got {len(arr)}")

    def _append(self, ds, arr):
        arr = np.asarray(arr)

        old = ds.shape[0]
        new = old + arr.shape[0]

        if len(ds.shape) == 1:
            ds.resize((new,))
        else:
            ds.resize((new, ds.shape[1]))

        ds[old:new] = arr

    # ------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(self.vector_id.shape[0])

    def dim(self) -> int:
        return int(self.emb_norm.shape[1]) if len(self.emb_norm.shape) > 1 else 0

    def __len__(self) -> int:
        return self.n_events

    # ------------------------------------------------------------
    # single-store streaming (NOT cross-slice)
    # ------------------------------------------------------------

    def iter_batches(self, batch_size: int = 8192):
        """
        Stream embeddings from this single event store.

        This is intentionally *not* cross-slice aware.

        Use ZarrEventStream for multi-slice FAISS construction.
        """

        n = self.n_events

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            yield (
                np.asarray(self.emb_norm[start:end], dtype=np.float32),
                np.asarray(self.vector_id[start:end], dtype=np.int64),
            )

    def iter_events(self, batch_size: int = 8192):
        """
        Full event view including provenance fields.
        Useful for debugging, analysis, and reconstruction.
        """

        n = self.n_events

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)

            yield {
                "emb_norm": self.emb_norm[start:end],
                "emb_raw": self.emb_raw[start:end],
                "vector_id": self.vector_id[start:end],
                "doc_id": self.doc_id[start:end],
                "token_idx": self.token_idx[start:end],
            }
