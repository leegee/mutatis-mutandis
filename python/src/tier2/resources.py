"""
tier2.resources

Construction of expensive, long-lived resources used by Tier 2:

- FAISS index loading (local / medium / broad per publication year)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger


def load_indices(paths_by_year, workers=6):
    """
    Load FAISS indices for every (year, scale) pair.

    Each year has independent local / medium / broad indices.
    Loading is isolated from analysis so failures are visible.
    """
    jobs = [
        (year, scale, path)
        for year, scales in paths_by_year.items()
        for scale, path in scales.items()
    ]

    logger.info(f"[tier2] loading {len(jobs)} FAISS indices")

    indexes = {
        year: {}
        for year in paths_by_year
    }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                EeboFaissIndex.load,
                path,
            ):
            (year, scale)
            for year, scale, path in jobs
        }

        for future in as_completed(futures):
            year, scale = futures[future]
            indexes[year][scale] = future.result()

    for year, scales in indexes.items():
        for scale, index in scales.items():
            if index.ntotal == 0:
                raise RuntimeError(f"Empty FAISS index: {year}/{scale}")

    return indexes
