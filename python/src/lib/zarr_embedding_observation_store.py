"""
ZarrEmbeddingObservationStore - Tier 1 contextual observation layer

Each row is a contextual observation event:

    event = (
        event_id,             # unique contextual observation identity
        concept_id,           # stable corpus token identity
        vector_id,            # lexical identity from corpus event log
        doc_id,               # document provenance
        token_idx,            # corpus position anchor
        window_id,            # transformer window start coordinate
        window_token_pos,     # token position within window
        emb_raw               # raw contextual embedding
    )

Core invariants
----------------

1. Postgres defines corpus truth.
2. concept_id identifies stable lexical occurrence in corpus space.
3. event_id identifies a single contextual embedding observation.
4. FAISS indexes event_id space ONLY.
5. vector_id is lexical identity only - NOT observation identity.
6. window_id is a first-class contextual coordinate.
7. window_token_pos preserves transformer positional structure.
8. One event_id MUST correspond to exactly one emitted vector.
9. No aggregation or semantic collapsing occurs in this layer.

Failure modes
-------------

- Keying FAISS by concept_id collapses contextual multiplicity.
- Keying FAISS by vector_id collapses contextual multiplicity.
- Removing window_id destroys contextual provenance.
- Removing window_token_pos destroys positional structure.
- Treating observations as corpus truth corrupts frequency analysis.
- Concurrent writers break append determinism.
"""

from __future__ import annotations

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrEmbeddingObservationStore:
    """
    Append-only contextual observation store.

    This layer records how transformer context windows
    observe lexical events from the corpus.
    """

    def __init__(self, path: str, dim: int):
        self.root = zarr.open_group(path, mode="a", zarr_version=2)
        self.dim = dim

        compressor = Blosc(
            cname="zstd",
            clevel=3,
            shuffle=Blosc.BITSHUFFLE
        )

        g = self.root.require_group("events")

        # contextual observation identity
        self.event_id = self._ds(
            g,
            "event_id",
            (),
            compressor,
            "int64"
        )

        # stable corpus token identity
        self.concept_id = self._ds(
            g,
            "concept_id",
            (),
            compressor,
            "int64"
        )

        # contextual embedding
        self.emb_raw = self._ds(
            g,
            "emb_raw",
            (dim,),
            compressor,
            "float32"
        )

        # corpus coordinates
        self.vector_id = self._ds(
            g,
            "vector_id",
            (),
            compressor,
            "int64"
        )

        self.token_idx = self._ds(
            g,
            "token_idx",
            (),
            compressor,
            "int64"
        )

        self.token = self._ds(
            g,
            "token",
            (),
            compressor,
            "U32"
        )

        self.doc_id = self._ds(
            g,
            "doc_id",
            (),
            compressor,
            "U32"
        )

        # contextual coordinates
        self.window_id = self._ds(
            g,
            "window_id",
            (),
            compressor,
            "int64"
        )

        self.window_token_pos = self._ds(
            g,
            "window_token_pos",
            (),
            compressor,
            "int32",
        )

    # dataset helper
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

    def append_events(
        self,
        event_id,
        concept_id,
        emb_raw,
        vector_id,
        doc_id,
        token_idx,
        token,
        window_id,
        window_token_pos,
    ):
        event_id = np.asarray(event_id, dtype=np.int64)
        concept_id = np.asarray(concept_id, dtype=np.int64)

        emb_raw = np.asarray(emb_raw, dtype=np.float32)

        vector_id = np.asarray(vector_id, dtype=np.int64)
        token_idx = np.asarray(token_idx, dtype=np.int64)

        token = np.asarray(token, dtype="U32")
        doc_id = np.asarray(doc_id, dtype="U32")

        window_id = np.asarray(window_id, dtype=np.int64)

        window_token_pos = np.asarray(
            window_token_pos,
            dtype=np.int32
        )

        n = event_id.shape[0]

        self._check(event_id, n)
        self._check(concept_id, n)

        self._check(emb_raw, n)

        self._check(vector_id, n)
        self._check(token_idx, n)

        self._check(token, n)
        self._check(doc_id, n)

        self._check(window_id, n)
        self._check(window_token_pos, n)

        self._append(self.event_id, event_id)
        self._append(self.concept_id, concept_id)

        self._append(self.emb_raw, emb_raw)

        self._append(self.vector_id, vector_id)
        self._append(self.token_idx, token_idx)

        self._append(self.token, token)
        self._append(self.doc_id, doc_id)

        self._append(self.window_id, window_id)
        self._append(self.window_token_pos, window_token_pos)

    def _check(self, arr, n):
        if len(arr) != n:
            raise ValueError(
                f"event size mismatch: expected {n}, got {len(arr)}"
            )

    def _append(self, ds, arr):
        arr = np.asarray(arr)

        old = ds.shape[0]
        new = old + arr.shape[0]

        if len(ds.shape) == 1:
            ds.resize((new,))
        else:
            ds.resize((new, ds.shape[1]))

        ds[old:new] = arr


    @property
    def n_events(self) -> int:
        return int(self.event_id.shape[0])

    def get_doc_ids(self) -> set[str]:
        if self.doc_id.shape[0] == 0:
            return set()
        return set(self.doc_id[:])

    def embedding_dim(self) -> int:
        if len(self.emb_raw.shape) <= 1:
            return 0

        return int(self.emb_raw.shape[1])

    def __len__(self) -> int:
        return self.n_events
