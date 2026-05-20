#!/usr/bin/env python
"""
tier1_5_build_faiss_index.py

Streaming FAISS construction over EEBO Zarr event logs.

Architecture
------------

Tier1 Zarr is a multi-slice append-only event store:

    ZARR_ROOT/tier1/<slice>/events/*

This builder constructs a global FAISS index by streaming:

    ZarrEventStream → FAISS (EeboFaissIndex)

Key invariants
--------------

1. Zarr is the only source of truth for embeddings + event IDs
2. FAISS stores only:
      - L2-normalised embedding vectors
      - stable vector_id keys
3. No full corpus materialisation
4. Cross-slice aggregation is streaming-only
"""

from __future__ import annotations

import numpy as np

from lib.eebo_config import ZARR_ROOT, INDEXES_DIR
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream


BATCH_SIZE = 8192


# ------------------------------------------------------------
# FAISS build
# ------------------------------------------------------------

def build_index(stream: ZarrEventStream) -> EeboFaissIndex:
    """
    Build FAISS index from streamed Zarr embeddings.

    Invariant:
        vector_id is globally unique across slices
    """

    index = None
    total = 0

    logger.info("[faiss-build] streaming Zarr event logs")

    for vecs, ids in stream.iter_embeddings(batch_size=BATCH_SIZE):

        if vecs is None or len(vecs) == 0:
            continue

        if index is None:
            dim = vecs.shape[1]
            index = EeboFaissIndex(dim=dim, exact=True)

        index.add(vecs, ids)
        total += len(ids)

    if index is None:
        raise RuntimeError("No embeddings found in Zarr event stream")

    logger.info(f"[faiss-build] complete ntotal={total}")

    return index


# ------------------------------------------------------------
# entry point
# ------------------------------------------------------------

def main():
    logger.info("[faiss-build] loading Zarr event stream")

    stream = ZarrEventStream(str(ZARR_ROOT / "tier1"))

    logger.info("[faiss-build] building FAISS index")

    index = build_index(stream)

    out_path = INDEXES_DIR / "faiss" / "tier1.index"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index.save(out_path)

    logger.info(f"[faiss-build] done → {out_path}")


if __name__ == "__main__":
    main()
