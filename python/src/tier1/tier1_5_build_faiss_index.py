#!/usr/bin/env python

"""
tier1_5_build_faiss_index.py

Builds per-year, per-scale FAISS indices (local / medium / broad) from the
Tier 1 observation store.

    Postgres (identity + text provenance)
        |
    Observation store (Zarr or Parquet)              <- read here via API
        |
    FAISS index (approximate geometric retrieval)    <- written here

This script owns index *construction and persistence* only. It does not
interpret embeddings, compute them, or reach back into Postgres.

The embedding source is an ObservationStream obtained from the observation
store factory. Backend selection is a CLI/config concern:

    --backend zarr      (default; historical path)
    --backend parquet   (hive-partitioned Parquet + DuckDB)

Partitioning
------------
Indices are partitioned by (year, scale) rather than one corpus-wide index
per scale. This bounds the size of any single FAISS index, keeps
incremental updates scoped to the years actually touched, and lets
downstream search restrict queries to a year range.

Within a year, the three scales (local/medium/broad) are always populated
in lockstep from the same event_id set — sync between them is treated as
an invariant and checked explicitly before any incremental build proceeds.

Memory strategy
---------------
The corpus is too large to hold every year's indices in memory for a
single streamed pass. service() processes years in bounded chunks
(year_chunk, default 20): for each chunk it builds/updates only that
chunk's indices, persists them to disk, and frees them before moving to
the next chunk.

iter_multi_scale_embeddings' optional year_filter lets a chunked pass skip
embedding reads for non-overlapping years. Zarr may supply a year_manifest
to accelerate that pre-check; Parquet returns an empty manifest and relies
on hive partitioning + SQL WHERE instead.

Modes
-----
- Fresh build: no existing per-year indices found, or --clear passed.
- Incremental: existing per-year indices are loaded, their ids() unioned
  into already_indexed, and build_indices() skips any event_id already
  present.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from lib.corpus_config import ZARR_PATH, MASKED_ZARR_PATH, faiss_index_paths
from lib.corpus_logging import logger
from lib.eebo_faiss import EeboFaissIndex

from observation_store_api import (
    ObservationStream,
    open_observation_stream,
)

# Register backends (import side-effect).
import lib.zarr_observation_backend  # noqa: F401
import lib.parquet_observation_backend  # noqa: F401

BATCH_SIZE = 8192
SCALES = ("local", "medium", "broad")
DEFAULT_BACKEND = "zarr"


def discover_years(masked: bool) -> list[int]:
    """
    Find which years already have on-disk indices, by globbing the 'medium'
    scale's directory for files matching the tier1_medium_<year>[_masked].faiss
    pattern and extracting the year.
    """
    medium_base = faiss_index_paths(masked)["medium"]  # tier1_medium[_masked].faiss
    masked_part = "_masked" if masked else ""
    pattern = f"tier1_medium_*{masked_part}.faiss"
    year_re = re.compile(rf"^tier1_medium_(\d+){masked_part}\.faiss$")

    years = []
    for p in medium_base.parent.glob(pattern):
        m = year_re.match(p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def resolve_year_range(
    stream: ObservationStream,
    year_manifest: Mapping[Any, np.ndarray],
) -> tuple[int, int]:
    """
    Determine (lo, hi) pub_year bounds for chunked iteration.

    Prefer year_manifest when the backend provides one (Zarr). Fall back to
    stream.year_bounds() for backends that return an empty manifest (Parquet).
    """
    nonzero = [y for y in year_manifest.values() if getattr(y, "size", 0)]
    if nonzero:
        lo = min(int(y.min()) for y in nonzero)
        hi = max(int(y.max()) for y in nonzero)
        return lo, hi
    return stream.year_bounds()


def build_indices(
    stream: ObservationStream,
    indices: dict[int, dict[str, EeboFaissIndex]] | None = None,
    already_indexed: set[int] | None = None,
    year_filter: set[int] | None = None,
    year_manifest: Mapping[Any, np.ndarray] | None = None,
) -> dict[int, dict[str, EeboFaissIndex]]:
    """
    Build/update per-year, per-scale FAISS indices from a single streamed
    pass over an ObservationStream.

    Each (year, scale) pair receives the same event_id set. FAISS stores
    only event geometry; event_id remains the stable semantic observation id.

    year_filter:
        If given, only events whose pub_year is in this set are indexed.
        Backends push this filter as far down as they can (Zarr array mask,
        Parquet SQL WHERE / hive partition prune).
    """
    if indices is None:
        indices = {}

    seen_stream_ids: set[int] = set()
    total = 0
    skipped = 0
    incremental = already_indexed is not None

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings by year")

    for (
        emb_local,
        emb_medium,
        emb_broad,
        obs_ids,
        pub_years,
    ) in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE,
        year_filter=year_filter,
        year_manifest=year_manifest,
    ):
        duplicates = [
            int(eid) for eid in obs_ids if int(eid) in seen_stream_ids
        ]
        if duplicates:
            raise RuntimeError(
                f"Duplicate event_ids from Tier1 stream: {duplicates[:20]}"
            )

        seen_stream_ids.update(int(eid) for eid in obs_ids)

        if len(obs_ids) == 0:
            continue

        logger.debug(
            "[faiss-build-debug] batch=%d years=%s events=%d",
            len(obs_ids),
            sorted(set(map(int, pub_years))),
            len(obs_ids),
        )

        per_scale = {
            "local": emb_local,
            "medium": emb_medium,
            "broad": emb_broad,
        }

        if incremental:
            new_mask = np.array(
                [int(eid) not in already_indexed for eid in obs_ids]
            )
            if not new_mask.any():
                skipped += len(obs_ids)
                continue

            skipped += int((~new_mask).sum())
            obs_ids = obs_ids[new_mask]
            pub_years = pub_years[new_mask]
            per_scale = {
                scale: emb[new_mask] for scale, emb in per_scale.items()
            }

        for year in np.unique(pub_years):
            year = int(year)
            year_mask = pub_years == year
            year_obs_ids = obs_ids[year_mask]

            if year not in indices:
                indices[year] = {}

            logger.debug(
                "[faiss-build-debug] adding year=%d count=%d",
                year,
                len(year_obs_ids),
            )

            for scale, emb in per_scale.items():
                year_emb = emb[year_mask]

                if scale not in indices[year]:
                    indices[year][scale] = EeboFaissIndex(
                        dim=year_emb.shape[1],
                        exact=True,
                    )

                indices[year][scale].add(year_emb, year_obs_ids)

                logger.debug(
                    "[faiss-build-debug] year=%d scale=%s ntotal=%d",
                    year,
                    scale,
                    indices[year][scale].ntotal,
                )

            total += len(year_obs_ids)

        if total % 100_000 < len(obs_ids):
            logger.info(
                "[faiss-build] indexed %d events so far across %d years...",
                total,
                len(indices),
            )

    if not indices:
        raise RuntimeError("No embeddings found in Tier1 observation store")

    logger.info(
        "[faiss-build] finished - added=%d skipped=%d years=%s",
        total,
        skipped,
        sorted(indices.keys()),
    )
    return indices


def service(
    *,
    stream: ObservationStream,
    masked: bool = False,
    clear: bool = False,
    year_chunk: int = 20,
):
    """
    Chunked FAISS build over an ObservationStream.

    Processes years in windows of `year_chunk` so peak memory stays bounded.
    """
    started = time.perf_counter()
    existing_years = set(discover_years(masked))

    if clear:
        EeboFaissIndex.wipe_faiss_index()
        existing_years = set()

    logger.info("[faiss-build] building year manifest (backend-specific)")
    year_manifest = stream.build_year_manifest()
    lo, hi = resolve_year_range(stream, year_manifest)
    logger.info("[faiss-build] year range %d–%d", lo, hi)

    saved = {}

    for chunk_start in range(lo, hi + 1, year_chunk):
        chunk_years = set(range(chunk_start, min(chunk_start + year_chunk, hi + 1)))
        logger.info(
            "[faiss-build] processing years %d–%d",
            min(chunk_years),
            max(chunk_years),
        )

        indices: dict[int, dict[str, EeboFaissIndex]] = {}
        already_indexed: set[int] = set()

        for year in chunk_years & existing_years:
            indices[year] = {}
            year_id_sets = {}
            for scale, path in faiss_index_paths(masked, year=year).items():
                idx = EeboFaissIndex.load(path)
                indices[year][scale] = idx
                year_id_sets[scale] = idx.ids()
            medium_ids = year_id_sets["medium"]
            for scale, ids in year_id_sets.items():
                if ids != medium_ids:
                    raise RuntimeError(
                        f"FAISS indices out of sync for year {year}. "
                        f"Rebuild with --clear."
                    )
            already_indexed |= medium_ids

        indices = build_indices(
            stream,
            indices=indices,
            already_indexed=already_indexed or None,
            year_filter=chunk_years,
            year_manifest=year_manifest,
        )

        saved.update(persist_indices(indices, masked))
        logger.info(
            "Chunk summary: %s",
            {y: indices[y]["medium"].ntotal for y in sorted(indices)},
        )
        del indices

    elapsed = time.perf_counter() - started
    return {
        "generated": "tier1_5_faiss_build",
        "summary": {"years": len(saved), "scales": list(SCALES)},
        "indices": saved,
        "elapsed_seconds": round(elapsed, 3),
    }


def persist_indices(indices, masked):
    saved = {}

    for year, scale_indices in indices.items():
        paths = faiss_index_paths(masked, year=year)
        expected = scale_indices["medium"].ids()
        saved[year] = {}

        for scale, idx in scale_indices.items():
            if idx.ids() != expected:
                raise RuntimeError(
                    f"FAISS divergence before save: year={year} scale={scale}"
                )
            path = paths[scale]
            path.parent.mkdir(parents=True, exist_ok=True)
            idx.save(path)
            saved[year][scale] = {
                "path": str(path),
                "ntotal": idx.ntotal,
            }

    return saved


def default_store_path(backend: str, masked: bool) -> Path:
    """
    Resolve the observation-store root for a backend when --store is omitted.

    Zarr uses the historical ZARR_PATH / MASKED_ZARR_PATH.
    Parquet defaults to a sibling directory named tier1_parquet[_masked].
    """
    if backend == "zarr":
        return Path(MASKED_ZARR_PATH if masked else ZARR_PATH)

    # parquet (and any future backend): derive from the zarr path's parent
    zarr_path = Path(MASKED_ZARR_PATH if masked else ZARR_PATH)
    suffix = "_masked" if masked else ""
    return zarr_path.parent / f"tier1_parquet{suffix}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build per-year multi-scale FAISS indices from the Tier 1 observation store."
    )
    p.add_argument("--clear", action="store_true", help="Wipe existing FAISS indices and rebuild")
    p.add_argument("--mask", action="store_true", help="Use masked corpus paths")
    p.add_argument(
        "--backend",
        choices=["zarr", "parquet"],
        default=DEFAULT_BACKEND,
        help=f"Observation store backend (default: {DEFAULT_BACKEND})",
    )
    p.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override observation store root path (default depends on --backend)",
    )
    p.add_argument(
        "--year-chunk",
        type=int,
        default=20,
        help="Number of years to hold in memory per pass (default: 20)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    masked = args.mask
    backend = args.backend

    store_path = Path(args.store) if args.store else default_store_path(backend, masked)

    logger.info(
        "[faiss-build] backend=%s store=%s masked=%s clear=%s",
        backend,
        store_path,
        masked,
        args.clear,
    )

    stream = open_observation_stream(backend, store_path)

    result = service(
        stream=stream,
        masked=masked,
        clear=args.clear,
        year_chunk=args.year_chunk,
    )

    logger.info("[faiss-build] complete: %s", result["summary"])


if __name__ == "__main__":
    main()
