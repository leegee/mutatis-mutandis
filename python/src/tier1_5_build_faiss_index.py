#!/usr/bin/env python
"""
tier1_5_build_faiss_index.py - Tier 1.5: FAISS Retrieval Index Construction

This module builds the global FAISS retrieval index over the Tier 1 semantic
observation store.

Tier 1 provides corpus-grounded observations containing aligned contextual
embeddings at multiple scales. Tier 1.5 derives a single retrieval embedding
from these representations and inserts it into a FAISS index for efficient
nearest-neighbour search.

Architecture
------------

Tier 1 observation store

    Observation
        ├── metadata
        ├── emb_local
        ├── emb_medium
        └── emb_broad

            │

            ▼

    Weighted ensemble embedding

            │

            ▼

    Global FAISS observation index

The ensemble embedding is currently computed as:

    0.25 × local
  + 0.50 × medium
  + 0.25 × broad

This weighting is an implementation choice rather than a semantic claim and
may evolve as retrieval quality is evaluated.

Purpose
-------
FAISS provides efficient approximate nearest-neighbour retrieval over the
Tier 1 observation space.

It does not perform:

- semantic interpretation
- concept induction
- clustering
- semantic drift analysis
- corpus modelling

It is purely a geometric retrieval layer.

Key invariants
--------------

1. Tier 1 is the sole source of truth
   - Embeddings are read directly from the Tier 1 observation store.

2. FAISS stores retrieval representations only
   - The index contains L2-normalised ensemble embeddings together with
     stable observation identifiers.

3. Observation identity is preserved
   - Each FAISS entry refers to exactly one Tier 1 observation.

4. Lexical identity is independent of retrieval identity
   - Multiple observations may share the same vector_id while differing in
     contextual representation.

5. Streaming construction
   - The index is built incrementally from streamed batches without loading
     the full corpus into memory.

6. Retrieval is observation-based
   - Neighbourhoods are computed over contextual observations rather than
     lexical types or inferred concepts.

Incremental updates
-------------------
Existing FAISS indices may be extended without rebuilding from scratch.
Previously indexed observation IDs are detected and skipped, allowing newly
generated Tier 1 observations to be appended efficiently.

Design intent
-------------
Tier 1.5 deliberately remains a thin infrastructure layer separating storage
from retrieval. Tier 1 defines the semantic observation space, while higher
tiers perform neighbourhood analysis, concept modelling, clustering and
diachronic investigation using the retrieval capabilities provided here.
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
