"""
lib/zarr_embedding_observation_store.py

ZarrEmbeddingObservationStore — Tier 1 contextual observation layer

Each row is a contextual observation event:

    event = (
        event_id,             # unique contextual observation identity
        concept_id,           # stable corpus token identity
        vector_id,            # lexical identity from corpus event log
        doc_id,               # document provenance
        token_idx,            # corpus position anchor        (int32)
        window_id,            # transformer window start coordinate
        window_token_pos,     # token position within window  (int32)
        token,                # surface form
        emb_raw,              # raw contextual embedding      (float32)
        quality_score,        # OCR quality [0.0-1.0]         (float32)
        ingestion_timestamp,  # unix epoch seconds            (uint32)
    )

Core invariants
---------------
1.  Postgres defines corpus truth.
2.  concept_id identifies stable lexical occurrence in corpus space.
3.  event_id identifies a single contextual embedding observation.
4.  FAISS indexes event_id space ONLY.
5.  vector_id is lexical identity only - NOT observation identity.
6.  window_id is a first-class contextual coordinate.
7.  window_token_pos preserves transformer positional structure.
8.  One event_id MUST correspond to exactly one emitted vector.
9.  No aggregation or semantic collapsing occurs in this layer.

Failure modes
-------------
- Keying FAISS by concept_id collapses contextual multiplicity.
- Keying FAISS by vector_id collapses contextual multiplicity.
- Removing window_id destroys contextual provenance.
- Removing window_token_pos destroys positional structure.
- Treating observations as corpus truth corrupts frequency analysis.
- Concurrent writers break append determinism.

Schema notes
------------
- token_idx is int32 throughout (corpus positions never exceed 2^31).
- quality_score is float32 in [0.0, 1.0]; -1.0 signals unknown quality.
- ingestion_timestamp is uint32 (unix epoch; valid until 2106).
"""

from __future__ import annotations

import time

import numpy as np
import zarr
from numcodecs import Blosc


_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


class ZarrEmbeddingObservationStore:
    """
    Append-only contextual observation store.

    Records how transformer context windows observe lexical events
    from the corpus.  One row per (token x window) observation.
    """

    def __init__(self, path: str, dim: int):
        self.root = zarr.open_group(path, mode="a", zarr_version=2)
        self.dim  = dim

        g = self.root.require_group("events")

        # --- identity ---------------------------------------------------
        self.event_id   = self._ds(g, "event_id",   (),     "int64")
        self.concept_id = self._ds(g, "concept_id", (),     "int64")

        # --- embedding --------------------------------------------------
        self.emb_raw    = self._ds(g, "emb_raw",    (dim,), "float32")

        # --- corpus coordinates -----------------------------------------
        self.vector_id        = self._ds(g, "vector_id",        (), "int64")
        self.token_idx        = self._ds(g, "token_idx",        (), "int32")
        self.token            = self._ds(g, "token",            (), "U32")
        self.doc_id           = self._ds(g, "doc_id",           (), "U32")

        # --- contextual coordinates -------------------------------------
        self.window_id        = self._ds(g, "window_id",        (), "int32")
        self.window_token_pos = self._ds(g, "window_token_pos", (), "int32")

        # --- quality / provenance ---------------------------------------
        # quality_score: float32 in [0.0, 1.0].  -1.0 = unknown.
        self.quality_score       = self._ds(g, "quality_score",       (), "float32")
        # ingestion_timestamp: unix epoch seconds as uint32.
        self.ingestion_timestamp = self._ds(g, "ingestion_timestamp", (), "uint32")

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append_events(
        self,
        event_id:               np.ndarray,
        concept_id:             np.ndarray,
        emb_raw:                np.ndarray,
        vector_id:              np.ndarray,
        doc_id:                 np.ndarray,
        token_idx:              np.ndarray,
        token:                  np.ndarray,
        window_id:              np.ndarray,
        window_token_pos:       np.ndarray,
        quality_score:          np.ndarray | None = None,
        ingestion_timestamp:    np.ndarray | None = None,
    ) -> None:
        """
        Append a batch of observation events.

        quality_score and ingestion_timestamp are optional:
        - quality_score defaults to -1.0 (unknown) if not supplied.
        - ingestion_timestamp defaults to the current unix time if not supplied.
        """
        event_id         = np.asarray(event_id,         dtype=np.int64)
        concept_id       = np.asarray(concept_id,       dtype=np.int64)
        emb_raw          = np.asarray(emb_raw,          dtype=np.float32)
        vector_id        = np.asarray(vector_id,        dtype=np.int64)
        token_idx        = np.asarray(token_idx,        dtype=np.int32)
        token            = np.asarray(token,            dtype="U32")
        doc_id           = np.asarray(doc_id,           dtype="U32")
        window_id        = np.asarray(window_id,        dtype=np.int32)
        window_token_pos = np.asarray(window_token_pos, dtype=np.int32)

        n = event_id.shape[0]

        if quality_score is None:
            quality_score = np.full(n, -1.0, dtype=np.float32)
        else:
            quality_score = np.asarray(quality_score, dtype=np.float32)

        if ingestion_timestamp is None:
            now = np.uint32(int(time.time()))
            ingestion_timestamp = np.full(n, now, dtype=np.uint32)
        else:
            ingestion_timestamp = np.asarray(ingestion_timestamp, dtype=np.uint32)

        # Validate all arrays share the same batch dimension
        for name, arr in [
            ("event_id",             event_id),
            ("concept_id",           concept_id),
            ("emb_raw",              emb_raw),
            ("vector_id",            vector_id),
            ("token_idx",            token_idx),
            ("token",                token),
            ("doc_id",               doc_id),
            ("window_id",            window_id),
            ("window_token_pos",     window_token_pos),
            ("quality_score",        quality_score),
            ("ingestion_timestamp",  ingestion_timestamp),
        ]:
            if len(arr) != n:
                raise ValueError(
                    f"Batch size mismatch for '{name}': "
                    f"expected {n}, got {len(arr)}"
                )

        self._append(self.event_id,             event_id)
        self._append(self.concept_id,           concept_id)
        self._append(self.emb_raw,              emb_raw)
        self._append(self.vector_id,            vector_id)
        self._append(self.token_idx,            token_idx)
        self._append(self.token,                token)
        self._append(self.doc_id,               doc_id)
        self._append(self.window_id,            window_id)
        self._append(self.window_token_pos,     window_token_pos)
        self._append(self.quality_score,        quality_score)
        self._append(self.ingestion_timestamp,  ingestion_timestamp)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_doc_ids(self) -> set[str]:
        if self.doc_id.shape[0] == 0:
            return set()
        return set(self.doc_id[:])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_events(self) -> int:
        return int(self.event_id.shape[0])

    def embedding_dim(self) -> int:
        if len(self.emb_raw.shape) <= 1:
            return 0
        return int(self.emb_raw.shape[1])

    def __len__(self) -> int:
        return self.n_events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _ds(
        g:            zarr.Group,
        name:         str,
        shape_suffix: tuple,
        dtype:        str,
    ) -> zarr.Array:
        if name in g:
            return g[name]

        shape  = (0,) + shape_suffix
        chunks = (4096,) if not shape_suffix else (4096, shape_suffix[0])

        return g.create_dataset(
            name,
            shape      = shape,
            chunks     = chunks,
            dtype      = dtype,
            compressor = _COMPRESSOR,
        )

    @staticmethod
    def _append(ds: zarr.Array, arr: np.ndarray) -> None:
        arr = np.asarray(arr)
        old = ds.shape[0]
        new = old + arr.shape[0]
        if len(ds.shape) == 1:
            ds.resize((new,))
        else:
            ds.resize((new, ds.shape[1]))
        ds[old:new] = arr
