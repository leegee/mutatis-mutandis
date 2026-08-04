"""
ZarrEmbeddingObservationStore - Tier 1 contextual observation layer

Each row is a contextual observation event

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

WIP
---
Now stores three embedding vectors per event:
    emb_local, emb_medium, emb_broad

This enables multi-scale semantic analysis while preserving the original
medium-scale embeddings for backward compatibility.
"""

from __future__ import annotations

import numpy as np
import zarr
from numcodecs import Blosc


class ZarrEmbeddingObservationStore:
    """
    Append-only store supporting multi-window ensemble embeddings.

    This layer records how transformer context windows observe lexical events from the corpus.
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
        self.event_id = self._ds( g, "event_id", (), compressor, "int64" )

        # stable corpus token identity
        self.concept_id = self._ds( g, "concept_id", (), compressor, "int64" )

        # Multi-scale contextual embeddings
        self.emb_local  = self._ds(g, "emb_local",  (self.dim,), compressor, "float32")
        self.emb_medium = self._ds(g, "emb_medium", (self.dim,), compressor, "float32")
        self.emb_broad  = self._ds(g, "emb_broad",  (self.dim,), compressor, "float32")

        # corpus coordinates
        self.vector_id = self._ds( g, "vector_id", (), compressor, "int64" )
        self.token_idx = self._ds( g, "token_idx", (), compressor, "int64" )
        self.token = self._ds( g, "token", (), compressor, "U64" )
        self.corpus = self._ds( g, "corpus", (), compressor, "U32" )
        self.doc_id = self._ds( g, "doc_id", (), compressor, "U64" )
        self.pub_year = self._ds( g, "pub_year", (), compressor, "int16" )

        # contextual coordinates
        self.window_id = self._ds( g, "window_id", (), compressor, "int64" )
        self.window_token_pos = self._ds( g, "window_token_pos", (), compressor, "int32", )


    # dataset
    def _ds(self, g, name, shape_suffix, compressor, dtype):
        if name in g:
            return g[name]

        # New field being added to a group that may already have data —
        # refuse to silently create a 0-length dataset that would desync
        # from siblings already on disk.
        existing_lengths = {k: g[k].shape[0] for k in g.array_keys()}
        if existing_lengths and any(n > 0 for n in existing_lengths.values()):
            raise RuntimeError(
                f"Cannot add new field '{name}' to non-empty store — existing "
                f"fields have data (e.g. {existing_lengths}), but '{name}' has "
                f"none. Backfill '{name}' explicitly for existing rows before "
                f"resuming writes, or this store will silently desync."
            )

        shape = (0,) + shape_suffix
        chunks = (4096,) if len(shape_suffix) == 0 else (4096, shape_suffix[0])
        return g.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)


    def append_events(
        self,
        event_id,
        concept_id,
        emb_local,
        emb_medium,
        emb_broad,
        vector_id,
        corpus,
        doc_id,
        pub_year,
        token_idx,
        token,
        window_id,
        window_token_pos,
    ):
        """
        Append events with three embedding scales: local, medium, and broad.
        """
        event_id = np.asarray(event_id, dtype=np.int64)
        concept_id = np.asarray(concept_id, dtype=np.int64)

        emb_local  = np.asarray(emb_local,  dtype=np.float32)
        emb_medium = np.asarray(emb_medium, dtype=np.float32)
        emb_broad  = np.asarray(emb_broad,  dtype=np.float32)

        vector_id = np.asarray(vector_id, dtype=np.int64)
        token_idx = np.asarray(token_idx, dtype=np.int64)

        token = np.asarray(token, dtype="U32")
        doc_id = np.asarray(doc_id, dtype="U32")
        corpus = np.asarray(corpus, dtype="U32")
        pub_year = np.asarray(pub_year, dtype=np.int16)

        window_id = np.asarray(window_id, dtype=np.int64)
        window_token_pos = np.asarray(window_token_pos, dtype=np.int32)

        n = len(event_id)

        # Validation
        if not (len(emb_local) == len(emb_medium) == len(emb_broad) == n):
            raise ValueError(f"Embedding arrays have mismatched lengths: "
                           f"local={len(emb_local)}, medium={len(emb_medium)}, broad={len(emb_broad)}, n={n}")

        self._check(event_id, n)
        self._check(concept_id, n)
        self._check(vector_id, n)
        self._check(token_idx, n)
        self._check(token, n)
        self._check(doc_id, n)
        self._check(corpus, n)
        self._check(pub_year, n)
        self._check(window_id, n)
        self._check(window_token_pos, n)

        # Append embeddings
        self._append(self.emb_local,  emb_local)
        self._append(self.emb_medium, emb_medium)
        self._append(self.emb_broad,  emb_broad)

        # Append metadata
        self._append(self.event_id, event_id)
        self._append(self.concept_id, concept_id)
        self._append(self.vector_id, vector_id)
        self._append(self.token_idx, token_idx)
        self._append(self.token, token)
        self._append(self.doc_id, doc_id)
        self._append(self.corpus, corpus)
        self._append(self.pub_year, pub_year)
        self._append(self.window_id, window_id)
        self._append(self.window_token_pos, window_token_pos)


    def _check(self, arr, n):
        """Validate array length."""
        if len(arr) != n:
            raise ValueError(f"event size mismatch: expected {n}, got {len(arr)}")


    def _append(self, ds, arr):
        """Append array to a Zarr dataset."""
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


    def get_doc_keys(self) -> set[tuple[str, str]]:
        if self.doc_id.shape[0] == 0:
            return set()

        if self.corpus.shape[0] != self.doc_id.shape[0]:
            raise ValueError(
                f"corpus/doc_id length mismatch: corpus={self.corpus.shape[0]}, "
                f"doc_id={self.doc_id.shape[0]} — store is corrupted"
            )

        return set(zip(self.corpus[:], self.doc_id[:]))


    def get_event_ids(self) -> set[int]:
        """
        Return the set of event_ids already present in the store.

        Used by incremental writers so rerunning on additional concepts only
        computes observations that have not yet been written.
        """
        if self.event_id.shape[0] == 0:
            return set()

        return set(map(int, self.event_id[:]))


    def validate_event_ids(self, ids):
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate event_ids in append batch")


    def embedding_dim(self) -> int:
        return int(self.emb_medium.shape[1])


    def __len__(self) -> int:
        return self.n_events
