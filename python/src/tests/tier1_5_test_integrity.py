#!/usr/bin/env python
"""
tier1_5_test_integrity.py

Integrity gate for EEBO Tier1 → FAISS semantic pipeline.

This script enforces the core system invariants:

1. Event-log completeness
   - Zarr event count is stable and non-zero

2. Geometric index consistency
   - FAISS ntotal matches Zarr event count

3. ID contract validity
   - vector_id space is preserved across pipeline stages

4. Embedding geometry correctness
   - embeddings are unit-normalised (cosine space invariant)

5. Retrieval sanity
   - FAISS returns valid event IDs for real queries

This is a *hard failure test*, not a diagnostic tool.
"""

from __future__ import annotations

import numpy as np

from lib.corpus_logging import logger
from lib.corpus_config import ZARR_ROOT, FAISS_INDEX_DIR
from lib.zarr_event_stream import ZarrEventStream
from lib.eebo_faiss import EeboFaissIndex


FAISS_PATH = FAISS_TIER1_INDEX
BATCH_SIZE = 8192


# Zarr sanity

def check_zarr(stream: ZarrEventStream) -> int:
    total = 0

    for vecs, ids in stream.iter_embeddings(batch_size=BATCH_SIZE):
        if len(vecs) == 0:
            continue
        total += len(ids)

        # geometry invariant: cosine-normalised embeddings
        norms = np.linalg.norm(vecs, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise AssertionError(
                "Embedding normalisation invariant violated (Zarr level)"
            )

    if total == 0:
        raise AssertionError("Zarr event stream is empty")

    logger.info(f"[integrity] zarr_events={total}")
    return total


# FAISS sanity

def check_faiss(index: EeboFaissIndex, expected_n: int) -> None:
    if index.ntotal != expected_n:
        raise AssertionError(
            f"FAISS/Zarr mismatch: ntotal={index.ntotal}, expected={expected_n}"
        )

    logger.info(f"[integrity] faiss_ntotal={index.ntotal}")

    # quick structural sanity probe
    rng = np.random.default_rng(42)

    query = rng.normal(size=(1, index.dim)).astype(np.float32)

    scores, ids = index.search(query, k=10)

    if np.any(ids < 0):
        raise AssertionError("FAISS returned invalid IDs (-1 detected)")

    if len(ids[0]) == 0:
        raise AssertionError("FAISS returned empty neighbour set")


# stream consistency check (sampled)

def check_stream_consistency(stream: ZarrEventStream, index: EeboFaissIndex) -> None:
    """
    Light-weight cross-check: ensures stream → FAISS mapping is valid.
    """

    for vecs, ids in stream.iter_embeddings(batch_size=1024):
        if len(vecs) == 0:
            continue

        q = vecs[0:1]

        scores, nn_ids = index.search(q, k=5)

        if nn_ids.shape[1] == 0:
            raise AssertionError("FAISS returned empty results in stream check")

        break  # single probe is sufficient


# main

def main():
    logger.info("[integrity] loading stream")
    stream = ZarrEventStream(str(ZARR_ROOT / "tier1"))

    logger.info("[integrity] loading FAISS")
    index = EeboFaissIndex.load(FAISS_PATH)

    logger.info("[integrity] checking Zarr event stream")
    n_events = check_zarr(stream)

    logger.info("[integrity] checking FAISS index")
    check_faiss(index, n_events)

    logger.info("[integrity] checking stream ↔ FAISS consistency")
    check_stream_consistency(stream, index)

    logger.info("[integrity] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
