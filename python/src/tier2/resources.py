"""
tier2.resources

Construction of expensive, long-lived resources used by Tier 2:

- FAISS index loading (local / medium / broad per publication year)
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.eebo_faiss import EeboFaissIndex
from lib.eebo_logging import logger


class LazyYearIndices(Mapping):
    """
    Dict-like (Mapping) view over per-year FAISS indices that loads a
    year's local/medium/broad indices only on first access, and can
    evict years that are no longer needed.

    Subclassing collections.abc.Mapping — rather than hand-rolling each
    dict method — gives correct, genuinely lazy implementations of
    keys(), values(), items(), get(), __contains__, etc. for free,
    all built on top of __getitem__ and __iter__ below. That matters
    because callers elsewhere in the codebase (e.g. ZarrEventLookup's
    attach_index, which does `next(iter(index.values()))`) expect this
    to behave like a real dict without needing to know it's lazy.

    This keeps peak memory proportional to the years actually touched
    by a run, rather than every year present in the corpus.
    """

    def __init__(self, paths_by_year, workers=1):
        self._paths_by_year = paths_by_year
        self._workers = workers
        self._loaded = {}

    def __iter__(self):
        return iter(self._paths_by_year)

    def __contains__(self, year):
        # Override Mapping's default, which would call __getitem__ and
        # thus trigger a load just to answer a membership check.
        return year in self._paths_by_year

    def __len__(self):
        return len(self._paths_by_year)

    def __getitem__(self, year):
        if year not in self._paths_by_year:
            raise KeyError(year)
        if year not in self._loaded:
            self._loaded[year] = self._load_year(year)
        return self._loaded[year]

    def _load_year(self, year):
        scales = self._paths_by_year[year]
        logger.info(f"[tier2] lazy-loading FAISS indices for year={year}")

        loaded = {}
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(EeboFaissIndex.load, path): scale
                for scale, path in scales.items()
            }
            for future in as_completed(futures):
                scale = futures[future]
                loaded[scale] = future.result()

        for scale, index in loaded.items():
            if index.ntotal == 0:
                raise RuntimeError(f"Empty FAISS index: {year}/{scale}")

        return loaded

    def evict(self, year):
        """
        Drop a year's indices from memory once a run no longer needs them.
        """
        self._loaded.pop(year, None)

    def preloaded_years(self):
        return set(self._loaded)


def load_indices(paths_by_year, workers=1):
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
