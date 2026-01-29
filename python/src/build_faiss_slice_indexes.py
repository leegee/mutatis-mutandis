#!/usr/bin/env python
"""
build_faiss_slice_indexes.py

Script mode to build FAISS slice indexes for all configured slices.
Delegates actual work to lib/faiss_slices.py.
"""

from lib.faiss_slices import build_all_slices
from lib.eebo_logging import logger

if __name__ == "__main__":
    logger.info("Starting FAISS slice index build")
    build_all_slices()
    logger.info("FAISS slice index build complete")
