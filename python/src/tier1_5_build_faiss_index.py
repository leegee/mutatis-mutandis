#!/usr/bin/env python
"""
tier1_5_build_faiss_index.py

Streaming FAISS construction over Tier 1 contextual observation stores.

Architecture
------------

Tier 1 is a multi-slice contextual observation layer:

    ZARR_ROOT/tier1/<slice>/events/*

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

2. FAISS stores only:
      - L2-normalised embedding vectors
      - stable observation IDs

3. vector_id is lexical identity, NOT embedding identity.

4. Multiple contextual observations may share the same vector_id.

5. Cross-slice aggregation is streaming-only.

6. No full-corpus materialisation occurs during index construction.

7. FAISS geometry operates over contextual observations, not corpus events.
"""

from __future__ import annotations

import shutil
import numpy as np

from lib.eebo_config import ZARR_ROOT, INDEXES_DIR
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream


BATCH_SIZE = 8192


# ---------------------------------------------------------------------
# FAISS construction
# ---------------------------------------------------------------------

def build_index(stream: ZarrEventStream) -> EeboFaissIndex:
    """
    Build a global FAISS index over Tier 1 contextual observations.

    Observation identity is defined by stream position, not lexical identity.

    Important:
        vector_id is NOT unique at the embedding level because
        overlapping transformer windows generate multiple contextual
        observations for the same corpus token occurrence.
    """

    index = None
    total = 0

    logger.info("[faiss-build] streaming Tier1 observation stores")

    for vecs, obs_ids in stream.iter_embeddings(batch_size=BATCH_SIZE):

        if vecs is None or len(vecs) == 0:
            continue

        if index is None:
            dim = vecs.shape[1]
            index = EeboFaissIndex(dim=dim, exact=True)

        index.add(vecs, obs_ids)

        total += len(obs_ids)

    if index is None:
        raise RuntimeError(
            "No embeddings found in Tier1 observation stream"
        )

    logger.info(f"[faiss-build] complete ntotal={total}")

    return index


def clear_faiss_output():
    path = INDEXES_DIR / "faiss"

    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("[faiss-build] loading Tier1 observation stream")
    stream = ZarrEventStream(str(ZARR_ROOT / "tier1"))

    logger.info("[faiss-build] clearing existing FAISS indexes")
    clear_faiss_output()

    logger.info("[faiss-build] building FAISS observation index")
    index = build_index(stream)

    out_path = INDEXES_DIR / "faiss" / "tier1.index"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index.save(out_path)
    logger.info(f"[faiss-build] done -> {out_path}")


if __name__ == "__main__":
    main()
