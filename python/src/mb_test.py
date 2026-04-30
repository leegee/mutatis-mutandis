#!/usr/bin/env python

import numpy as np
import json
from collections import Counter

from lib.vector_store import load_id_vectors
from lib.mb_paths import faiss_slice_path
from lib.FaissIndex import FaissIndex
from lib.eebo_logging import logger
from lib.eebo_config import SLICES


def test_vector_store(slice_id: str):
    """
    Validates core invariants:

    1. NPZ vectors exist and are aligned
    2. id_to_pos is bijective
    3. FAISS index loads for same slice
    4. FAISS size matches vector store size (soft check)
    """

    logger.info(f"Testing slice={slice_id}")

    vecs, id_to_pos, ids = load_id_vectors(slice_id)

    # structural integrity of vector store
    if vecs.shape[0] != len(ids):
        raise ValueError(
            f"Vector store mismatch: vecs={vecs.shape[0]} ids={len(ids)}"
        )

    # Test
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise ValueError(
            f"Duplicate token_occurrence_id detected: "
            f"{len(dupes)} duplicated IDs (sample={dupes[:10]})"
        )

    if min(id_to_pos.values()) != 0:
        raise ValueError("id_to_pos not zero-indexed")

    if max(id_to_pos.values()) != len(ids) - 1:
        raise ValueError("id_to_pos not contiguous")

    # 2FAISS consistency check (soft dependency)
    index = FaissIndex.load(str(faiss_slice_path(slice_id)))

    faiss_size = index._index.ntotal

    if faiss_size != len(ids):
        logger.warning(
            f"FAISS/NPZ mismatch slice={slice_id} "
            f"faiss={faiss_size} npz={len(ids)}"
        )

    # 3sample reconstruction sanity (NOT FAISS reconstruct)
    # we only test vector identity consistency via mapping layer

    sample_ids = ids[: min(50, len(ids))]

    for tid in sample_ids:
        pos = id_to_pos[tid]

        if pos >= vecs.shape[0]:
            raise ValueError(f"Invalid id_to_pos mapping for {tid}")

        v = vecs[pos]

        if not np.isfinite(v).all():
            raise ValueError(f"Non-finite vector for id={tid}")

        if np.linalg.norm(v) == 0:
            logger.warning(f"Zero vector detected id={tid}")

    logger.info(f"OK slice={slice_id} vectors={len(ids)} faiss={faiss_size}")


def main():
    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        test_vector_store(slice_id)


if __name__ == "__main__":
    main()
