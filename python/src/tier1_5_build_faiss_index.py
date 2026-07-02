#!/usr/bin/env python
"""
tier1_5_build_faiss_index.py

Streaming FAISS construction over Tier 1 contextual observation stores.

Architecture
------------

Tier 1 is a single contextual observation store:

    ZARR_ROOT/tier1/events/*

Each stored row represents a contextual observation of a corpus token
under a specific transformer window.

This builder constructs a global FAISS geometry index by streaming:

    Tier1 Observation Store
        ->
    FAISS observation-space index

FAISS is retrieval infrastructure only.

It does NOT define:
    - semantic meaning
    - concepts
    - drift
    - clusters
    - fields

It provides approximate nearest-neighbour geometry over contextual
observations.

Key invariants
--------------

1. Tier 1 observation stores are the sole source of truth for embeddings.

2. FAISS stores only L2-normalised embedding vectors and stable observation IDs: it is intended as simply a retrieval layer and not a semantic model or clustering system.

3. vector_id is lexical identity, NOT embedding identity.

4. Multiple contextual observations may share the same vector_id.

5. No full-corpus materialisation occurs during index construction.

6. FAISS geometry operates over contextual observations, not corpus events.

WIP
---
Now builds FAISS using the ensemble of multi-window embeddings.

"""

from __future__ import annotations

import argparse
import numpy as np

from lib.eebo_config import ZARR_ROOT, FAISS_INDEX_DIR, FAISS_TIER1_INDEX
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream


BATCH_SIZE = 8192


def build_index(
    stream: ZarrEventStream,
    index: EeboFaissIndex | None = None,
    already_indexed: set[int] | None = None,
) -> EeboFaissIndex:
    """
    Build or incrementally update FAISS index using multi-window ensemble embeddings.
    """
    total = 0
    skipped = 0
    incremental = already_indexed is not None

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings")

    for emb_local, emb_medium, emb_broad, obs_ids in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE
    ):
        if len(obs_ids) == 0:
            continue

        # Compute ensemble embedding (weighted average)
        ensemble = (
            0.25 * emb_local +
            0.50 * emb_medium +
            0.25 * emb_broad
        )

        if index is None:
            dim = ensemble.shape[1]
            index = EeboFaissIndex(dim=dim, exact=True)

        if incremental:
            new_mask = np.array([int(i) not in already_indexed for i in obs_ids])
            if not new_mask.any():
                skipped += len(obs_ids)
                continue

            ensemble = ensemble[new_mask]
            obs_ids = obs_ids[new_mask]
            skipped += (~new_mask).sum()

        try:
            index.add(ensemble, obs_ids)
            total += len(obs_ids)
            if total % 100_000 == 0:
                logger.info(f"[faiss-build] indexed {total:,} events so far...")
        except Exception as e:
            logger.error(f"[faiss-build] add failed: {e}", exc_info=True)
            raise

    if index is None:
        raise RuntimeError("No embeddings found in Tier1 observation store")

    logger.info(f"[faiss-build] finished - added={total:,} skipped={skipped:,}")
    return index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true",
                   help="Wipe existing FAISS index and rebuild from scratch")
    return p.parse_args()


def main():
    args = parse_args()

    stream = ZarrEventStream(str(ZARR_ROOT / "tier1"))

    if args.clear or not FAISS_TIER1_INDEX.is_file():
        if args.clear:
            logger.info("[faiss-build] clearing existing FAISS index")
            EeboFaissIndex.wipe_faiss_index(FAISS_TIER1_INDEX.parent)

        logger.info("[faiss-build] building FAISS observation index from scratch")
        index = build_index(stream)          # always fresh when --clear or no index
    else:
        logger.info("[faiss-build] incremental mode — loading existing index")
        index = EeboFaissIndex.load(FAISS_TIER1_INDEX)
        already_indexed = index.ids()
        logger.info(f"[faiss-build] existing index ntotal={len(already_indexed)}")
        index = build_index(stream, index=index, already_indexed=already_indexed)

    FAISS_TIER1_INDEX.parent.mkdir(parents=True, exist_ok=True)
    index.save(FAISS_TIER1_INDEX)
    logger.info(f"[faiss-build] done -> {FAISS_TIER1_INDEX}")

if __name__ == "__main__":
    main()
