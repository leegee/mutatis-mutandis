"""
ZarrEmbeddingEventLog

True event-store for EEBO embedding inference.

Each stored record is a single atomic semantic event:

    event = (
        vector_id,     # stable lexical identity from Postgres
        doc_id,        # document provenance
        token_idx,     # position in document
        emb_norm,      # contextualised + normalised embedding
        emb_raw        # unnormalised embedding (model space)
    )

Design principles:

1. Event atomicity
   - Each row is independently meaningful
   - No semantic dependence on neighbouring rows

2. No implicit alignment
   - Column arrays are strictly parallel by construction
   - Integrity enforced at append-time

3. Reconstructability
   - Any subset of events can be rehydrated into sequences
   - Suitable for drift analysis, clustering, or temporal slicing

4. Append-only semantics
   - Zarr is treated as an immutable log structure
   - No in-place mutation of event semantics

This structure prioritises interpretability and downstream analytical
flexibility over storage compactness.
"""

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrEmbeddingEventLog:

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

    def append_events(self, emb_norm, emb_raw, vector_id, doc_id, token_idx):
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

    def __len__(self):
        return self.vector_id.shape[0]
