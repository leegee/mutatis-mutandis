"""
lib/zarr_event_stream.py

Streaming abstraction over sharded Tier 1 Zarr observation stores.

Role
----
Provides deterministic batch streaming of:

    (embeddings, event_ids)

for FAISS ingestion and any other consumer that needs to iterate
observations without materialising the full corpus.

Shard awareness
---------------
ZarrEventStream now accepts an explicit list of shard paths rather than
a single root directory.  ShardResolver.all_shards() is the canonical
way to produce that list.  This separates path resolution from streaming
logic cleanly.

Filtering
---------
Optional corpus_id and strategy filters are passed at construction time
and forwarded to ShardResolver.all_shards().  Callers that want a single
shard just pass a one-element list directly.

Lookup table warning
--------------------
_build_lookup() materialises a full {event_id -> token/doc} dict in memory.
At EEBO+ECCO scale this will be tens of millions of entries and may exhaust
RAM.  It is retained here for compatibility but callers should prefer direct
Zarr slice reads for production lookup.  A future ZarrEventLookup class
backed by on-disk sorted arrays is the intended replacement.

Invariants
----------
- event_id is the stable, globally unique observation identity.
- vector_id is lexical identity only — NOT used as a FAISS key.
- Schema mismatches fail loudly; missing datasets are not silently skipped.
- Batch-level streaming only; no full-corpus materialisation in iter_embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr

from lib.eebo_logging import logger
from lib.shard_resolver import ShardResolver
from lib.window_strategy import WindowStrategy


class ZarrEventStream:

    EXPECTED_GROUP   = "events"
    EXPECTED_EMB_KEY = "emb_raw"
    EXPECTED_ID_KEY  = "event_id"

    def __init__(
        self,
        shard_paths: list[Path] | None = None,
        resolver:    ShardResolver     | None = None,
        corpus_id:   str               | None = None,
        strategy:    WindowStrategy    | None = None,
    ):
        """
        Parameters
        ----------
        shard_paths:
            Explicit list of shard directories to stream.  Takes priority
            over resolver-based enumeration.  Pass a one-element list to
            stream a single shard.

        resolver:
            ShardResolver instance used to enumerate shards when
            shard_paths is not supplied.  If both are None a default
            ShardResolver() is constructed.

        corpus_id:
            Optional corpus filter forwarded to resolver.all_shards().

        strategy:
            Optional strategy filter forwarded to resolver.all_shards().
        """
        if shard_paths is not None:
            self._shard_paths = [Path(p) for p in shard_paths]
        else:
            r = resolver or ShardResolver()
            self._shard_paths = r.all_shards(
                corpus_id = corpus_id,
                strategy  = strategy,
            )

        if not self._shard_paths:
            logger.warning("[stream] no shard paths found — stream will be empty")

        self._token_by_id: dict[int, str] | None = None
        self._doc_by_id:   dict[int, str] | None = None

    # ------------------------------------------------------------------
    # Shard enumeration
    # ------------------------------------------------------------------

    @property
    def shard_paths(self) -> list[Path]:
        return list(self._shard_paths)

    # ------------------------------------------------------------------
    # Primary streaming interface
    # ------------------------------------------------------------------

    def iter_embeddings(self, batch_size: int = 8192):
        """
        Yield batches of (vecs, event_ids) across all shards.

            vecs:      (batch, dim)  float32
            event_ids: (batch,)      int64

        Shards are streamed sequentially; no cross-shard state is held
        between yields.
        """
        for shard_path in self._shard_paths:
            yield from self._stream_shard(shard_path, batch_size)

    def iter_shard_embeddings(self, batch_size: int = 8192):
        """
        Like iter_embeddings but yields (shard_path, vecs, event_ids),
        allowing callers that write one FAISS index per shard to track
        which shard each batch belongs to.
        """
        for shard_path in self._shard_paths:
            for vecs, ids in self._stream_shard(shard_path, batch_size):
                yield shard_path, vecs, ids

    # ------------------------------------------------------------------
    # Lookup helpers  (see memory warning in module docstring)
    # ------------------------------------------------------------------

    def token(self, event_id: int) -> str | None:
        self._build_lookup()
        return self._token_by_id.get(int(event_id))

    def doc_id(self, event_id: int) -> str | None:
        self._build_lookup()
        return self._doc_by_id.get(int(event_id))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stream_shard(self, shard_path: Path, batch_size: int):
        g = zarr.open_group(str(shard_path), mode="r")

        if self.EXPECTED_GROUP not in g:
            raise KeyError(
                f"Missing '{self.EXPECTED_GROUP}' group in {shard_path}"
            )

        group = g[self.EXPECTED_GROUP]

        if self.EXPECTED_EMB_KEY not in group:
            raise KeyError(
                f"Missing embeddings key '{self.EXPECTED_EMB_KEY}' "
                f"in {shard_path}"
            )

        if self.EXPECTED_ID_KEY not in group:
            raise KeyError(
                f"Missing event_id key in {shard_path}"
            )

        emb  = group[self.EXPECTED_EMB_KEY]
        eids = group[self.EXPECTED_ID_KEY]
        n    = eids.shape[0]

        if n == 0:
            return

        for start in range(0, n, batch_size):
            end  = min(start + batch_size, n)
            vecs = np.asarray(emb[start:end],  dtype=np.float32)
            ids  = np.asarray(eids[start:end], dtype=np.int64)

            if len(vecs) != len(ids):
                raise ValueError(
                    f"Embedding/id mismatch in {shard_path}: "
                    f"{len(vecs)} vs {len(ids)}"
                )

            yield vecs, ids

    def _build_lookup(self) -> None:
        if self._token_by_id is not None:
            return

        logger.warning(
            "[stream] _build_lookup() is materialising a full "
            "event_id->token/doc dict in memory.  At EEBO+ECCO scale "
            "this may exhaust RAM.  Prefer direct Zarr slice reads for "
            "production lookups."
        )

        token_map: dict[int, str] = {}
        doc_map:   dict[int, str] = {}

        for shard_path in self._shard_paths:
            g = zarr.open_group(str(shard_path), mode="r")

            if self.EXPECTED_GROUP not in g:
                continue

            group = g[self.EXPECTED_GROUP]

            if self.EXPECTED_ID_KEY not in group:
                raise KeyError(f"Missing event_id in {shard_path}")

            eids   = group[self.EXPECTED_ID_KEY][:]
            docs   = group["doc_id"][:] if "doc_id" in group else None
            tokens = group["token"][:]  if "token"  in group else None

            for i, eid in enumerate(eids):
                eid = int(eid)
                if docs   is not None:
                    doc_map[eid]   = str(docs[i])
                if tokens is not None:
                    token_map[eid] = str(tokens[i])

        self._token_by_id = token_map
        self._doc_by_id   = doc_map

        logger.info(f"[stream] lookup built: {len(token_map)} events")
