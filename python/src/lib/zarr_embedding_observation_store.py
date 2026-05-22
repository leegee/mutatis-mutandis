"""
ZarrEmbeddingObservationStore - Tier 1 contextual observation layer

This module defines a single-slice, append-only store of contextual observations
derived from a corpus event log.

It does NOT store corpus events directly.

Instead, it stores model-mediated observations of those events under
overlapping transformer windows.

Each row is a contextual observation event:

    event = (
        vector_id,            # lexical identity (from corpus event log)
        doc_id,               # document provenance
        token_idx,            # corpus position (anchor in document)
        window_id,            # transformer window start coordinate
        window_token_pos,     # token position within window
        emb_raw               # raw contextual embedding (model space)
    )

Core invariants
----------------

1. Corpus event log defines ground truth token occurrences (Postgres).
2. This store records observations under contextual windows.
3. window_id is a first-class sampling coordinate (not metadata).
4. window_token_pos preserves intra-window positional structure.
5. No aggregation, clustering, or normalisation is performed here.
6. Column alignment is structural, not semantic.

Failure modes
-------------

- Removing window_id collapses contextual multiplicity.
- Removing window_token_pos destroys intra-window structure.
- Treating this store as corpus truth leads to incorrect frequency analysis.
- Concurrent writers break determinism.
"""

from __future__ import annotations

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrEmbeddingObservationStore:
    """
    Append-only observation store over Zarr arrays.

    This is a measurement layer over a corpus event log.

    It records how a model observes token events under contextual windows.
    """

    def __init__(self, path: str, dim: int):
        self.root = zarr.open_group(path, mode="a", zarr_version=2)
        self.dim = dim

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        g = self.root.require_group("events")

        # embedding
        self.emb_raw = self._ds(g, "emb_raw", (dim,), compressor, "float32")

        # corpus identity
        self.vector_id = self._ds(g, "vector_id", (), compressor, "int64")
        self.token_idx = self._ds(g, "token_idx", (), compressor, "int64")
        self.token = self._ds(g, "token", (), compressor, "U32")
        self.doc_id = self._ds(g, "doc_id", (), compressor, "U32")

        # contextual coordinates
        self.window_id = self._ds(g, "window_id", (), compressor, "int64")
        self.window_token_pos = self._ds(
            g,
            "window_token_pos",
            (),
            compressor,
            "int32",
        )

    # ------------------------------------------------------------
    # dataset helper
    # ------------------------------------------------------------

    def _ds(self, g, name, shape_suffix, compressor, dtype):
        if name in g:
            return g[name]

        shape = (0,) + shape_suffix

        chunks = (4096,)
        if len(shape_suffix) > 0:
            chunks = (4096, shape_suffix[0])

        return g.create_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )

    # ------------------------------------------------------------
    # write path
    # ------------------------------------------------------------

    def append_events(
        self,
        emb_raw,
        vector_id,
        doc_id,
        token_idx,
        token,
        window_id,
        window_token_pos,
    ):
        emb_raw = np.asarray(emb_raw, dtype=np.float32)
        vector_id = np.asarray(vector_id, dtype=np.int64)
        token_idx = np.asarray(token_idx, dtype=np.int64)
        token = np.asarray(token, dtype="U32")
        doc_id = np.asarray(doc_id, dtype="U32")
        window_id = np.asarray(window_id, dtype=np.int64)
        window_token_pos = np.asarray(window_token_pos, dtype=np.int32)

        n = vector_id.shape[0]

        self._check(emb_raw, n)
        self._check(vector_id, n)
        self._check(token_idx, n)
        self._check(token, n)
        self._check(doc_id, n)
        self._check(window_id, n)
        self._check(window_token_pos, n)

        self._append(self.emb_raw, emb_raw)
        self._append(self.vector_id, vector_id)
        self._append(self.token_idx, token_idx)
        self._append(self.token, token)
        self._append(self.doc_id, doc_id)
        self._append(self.window_id, window_id)
        self._append(self.window_token_pos, window_token_pos)

    # ------------------------------------------------------------
    # safety
    # ------------------------------------------------------------

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

    def embedding_dim(self) -> int:
        return int(self.emb_raw.shape[1]) if len(self.emb_raw.shape) > 1 else 0

    def __len__(self) -> int:
        return self.n_events
