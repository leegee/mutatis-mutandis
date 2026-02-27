#!/usr/bin/env python
"""
Unified pipeline for generating slice embeddings (unaligned or aligned)
and building FAISS indexes per slice.
"""

from __future__ import annotations
import os
import argparse
from typing import Optional

import numpy as np

from lib.eebo_config import SLICES, SLICES_DIR, FASTTEXT_PARAMS, FAISS_INDEX_DIR
from lib.eebo_logging import logger
from lib.eebo_embeddings import generate_embeddings_per_model
from align import load_aligned_vectors, align_to_reference
from lib.faiss_slices import build_index_for_slice

USE_ALIGNED_ENV = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS", "0") == "1"


def generate_and_store_embeddings(
    slice_range: tuple[int, int],
    use_aligned: bool,
    reference_slice_id: Optional[str] = None
) -> dict[str, np.ndarray]:
    """Generate embeddings for a slice, optionally align to reference slice."""
    start, end = slice_range
    if use_aligned:
        slice_id = f"{start}-{end}"
        logger.info(f"Loading and aligning slice {slice_id} to reference {reference_slice_id}")
        embeddings = load_aligned_vectors(slice_id)
        if reference_slice_id is not None:
            embeddings = align_to_reference(embeddings, reference_slice_id)
    else:
        slice_file = SLICES_DIR / f"{start}-{end}.txt"
        if not slice_file.exists():
            logger.warning(f"Slice file {slice_file} missing, skipping")
            return {}
        embeddings = generate_embeddings_per_model(slice_file)
    return embeddings


def build_all_slices(use_aligned: bool = False, reference_slice_id: Optional[str] = None):
    """Generate embeddings and build FAISS indexes for all slices."""
    for slice_range in SLICES:
        embeddings = generate_and_store_embeddings(
            slice_range, use_aligned, reference_slice_id
        )
        if embeddings:
            build_index_for_slice(slice_range, embeddings)
    logger.info("All slices processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings and build FAISS indexes per slice"
    )
    parser.add_argument(
        "--aligned", action="store_true",
        help="Align all slices to reference slice before building FAISS indexes"
    )
    parser.add_argument(
        "--reference", type=str, default="1625-1629",
        help="Reference slice ID for alignment"
    )
    args = parser.parse_args()

    # Environment variable takes precedence if set
    use_aligned = USE_ALIGNED_ENV or args.aligned

    build_all_slices(
        use_aligned=use_aligned,
        reference_slice_id=args.reference
    )


if __name__ == "__main__":
    main()