#!/usr/bin/env python

"""
tier1_5_build_faiss_index.py

Builds per-year, per-scale FAISS indices (local / medium / broad) from the
Tier1 Zarr event log.

    Postgres (identity + text provenance)
        |
    Zarr event log (canonical semantic events)          <- read here
        |
    FAISS index (approximate geometric retrieval)        <- written here

This script owns index *construction and persistence* only. It does not
interpret embeddings, compute them, or reach back into Postgres - see
eebo_faiss.py for the FAISS wrapper itself and zarr_event_stream.py for the
streaming source.

Partitioning
------------
Indices are partitioned by (year, scale) rather than one corpus-wide index
per scale. This bounds the size of any single FAISS index, keeps
incremental updates scoped to the years actually touched, and lets
downstream search (multiscale_search in eebo_faiss.py) restrict queries to
a year range instead of always searching the whole corpus.

Within a year, the three scales (local/medium/broad) are always populated
in lockstep from the same event_id set - sync between them is treated as
an invariant and checked explicitly (see service()'s id-set comparison)
before any incremental build is allowed to proceed.

Memory strategy
----------------
The corpus is too large to hold every year's indices in memory for a
single streamed pass. service() processes years in bounded chunks
(year_chunk, default 20): for each chunk it builds/updates only that
chunk's indices, persists them to disk, and frees them before moving to
the next chunk. This trades additional store re-reads (one pass over the
Zarr store per chunk) for a bounded memory footprint - necessary because
a single-pass, all-years-resident build was observed to exhaust available
memory (std::bad_alloc from FAISS's internal buffer growth) partway
through a full corpus run.

iter_multi_scale_embeddings' optional year_filter (see zarr_event_stream.py)
lets a chunked pass skip embedding reads entirely for stores/batches with
no overlap, using the cheap int16 pub_year array as a pre-check - so the
repeated-pass cost is mitigated but not eliminated. This is a deliberate
throughput/memory trade-off, not an accident. TODO: revisit when the store
layout guarantees year-locality and whole stores can be skipped outright.

Modes
-----
- Fresh build: no existing per-year indices found, or --clear passed.
  Every matched (chunk, year, scale) index is built from scratch.
- Incremental: existing per-year indices are loaded, their ids() unioned
  into already_indexed, and build_indices() skips any event_id already
  present. New events are added to the loaded indices in place before
  re-persisting.

Entry points: main() parses --clear/--mask, opens a ZarrEventStream over
the (optionally masked) Tier1 store, and delegates to service().
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np

from lib.eebo_config import ZARR_PATH, MASKED_ZARR_PATH, faiss_index_paths
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream

BATCH_SIZE = 8192
SCALES = ("local", "medium", "broad")


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


def build_indices(
    stream: ZarrEventStream,
    indices: dict[int, dict[str, EeboFaissIndex]] | None = None,
    already_indexed: set[int] | None = None,
    year_filter: set[int] | None = None,
) -> dict[int, dict[str, EeboFaissIndex]]:
    """
    Build/update per-year, per-scale FAISS indices from a single streamed
    pass. Each (year, scale) pair gets its own EeboFaissIndex, so downstream
    fusion can happen both across scales and within/across year ranges.

    year_filter:
        If given, only events whose pub_year is in this set are indexed.
        Everything else is skipped before any per-scale slicing happens.
        Used by service() to process the corpus in bounded year-chunks so
        we never hold every year's indices in memory at once.
    """
    total = 0
    skipped = 0
    incremental = already_indexed is not None
    indices = indices or {}

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings by year")

    for emb_local, emb_medium, emb_broad, obs_ids, pub_years in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE, year_filter=year_filter
    ):
        if len(obs_ids) == 0:
            continue

        per_scale = {"local": emb_local, "medium": emb_medium, "broad": emb_broad}

        if incremental:
            new_mask = np.array([int(i) not in already_indexed for i in obs_ids])
            if not new_mask.any():
                skipped += len(obs_ids)
                continue
            skipped += (~new_mask).sum()
            obs_ids = obs_ids[new_mask]
            pub_years = pub_years[new_mask]
            per_scale = {s: e[new_mask] for s, e in per_scale.items()}

        unique_years = np.unique(pub_years)

        try:
            for year in unique_years:
                year = int(year)
                year_mask = pub_years == year
                year_obs_ids = obs_ids[year_mask]

                if year not in indices:
                    indices[year] = {}

                for scale, emb in per_scale.items():
                    year_emb = emb[year_mask]
                    if scale not in indices[year]:
                        indices[year][scale] = EeboFaissIndex(dim=year_emb.shape[1], exact=True)
                    indices[year][scale].add(year_emb, year_obs_ids)

                total += len(year_obs_ids)

            if total % 100_000 < len(obs_ids):
                logger.info(f"[faiss-build] indexed {total:,} events so far "
                            f"across {len(indices)} years...")
        except Exception as e:
            logger.error(f"[faiss-build] add failed: {e}", exc_info=True)
            raise

    if not indices:
        raise RuntimeError("No embeddings found in Tier1 observation store")

    logger.info(f"[faiss-build] finished - added={total:,} skipped={skipped:,} "
                f"years={sorted(indices.keys())}")
    return indices



def service(*, stream, masked: bool = False, clear: bool = False, year_chunk: int = 20):
    started = time.perf_counter()
    existing_years = set(discover_years(masked))

    if clear:
        for year in existing_years:
            for scale, path in faiss_index_paths(masked, year=year).items():
                EeboFaissIndex.wipe_faiss_index(path)
        existing_years = set()

    lo, hi = stream.year_bounds()
    saved = {}

    for chunk_start in range(lo, hi + 1, year_chunk):
        chunk_years = set(range(chunk_start, min(chunk_start + year_chunk, hi + 1)))
        logger.info(f"[faiss-build] processing years {min(chunk_years)}-{max(chunk_years)}")

        indices = {}
        already_indexed = set()

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
                    raise RuntimeError(f"FAISS indices out of sync for year {year}. Rebuild with --clear.")
            already_indexed |= medium_ids

        indices = build_indices(
            stream,
            indices=indices,
            already_indexed=already_indexed or None,
            year_filter=chunk_years,
        )

        saved.update(persist_indices(indices, masked))
        del indices   # <-- release this chunk's memory before the next pass

    elapsed = time.perf_counter() - started
    return {
        "generated": "tier1_5_faiss_build",
        "summary": {"years": len(saved), "scales": list(SCALES)},
        "indices": saved,
        "elapsed_seconds": round(elapsed, 3),
    }


def persist_indices(
    indices,
    masked,
):
    saved = {}

    for year, scale_indices in indices.items():
        paths = faiss_index_paths(masked, year=year)

        saved[year] = {}

        for scale, idx in scale_indices.items():
            path = paths[scale]
            path.parent.mkdir(parents=True, exist_ok=True)
            idx.save(path)

            saved[year][scale] = {
                "path": str(path),
                "ntotal": idx.ntotal,
            }

    return saved


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--mask", action="store_true")
    return p.parse_args()


def old_main():
    args = parse_args()
    masked = args.mask
    zarr_path = MASKED_ZARR_PATH if args.mask else ZARR_PATH

    logger.info(f"[faiss-build] mode={'unmask' if args.mask else 'maskeded'} zarr={zarr_path}")

    stream = ZarrEventStream(str(zarr_path))
    existing_years = discover_years(masked)

    if args.clear:
        for year in existing_years:
            for scale, path in faiss_index_paths(masked, year=year).items():
                logger.info(f"[faiss-build] clearing existing {scale}/{year} index")
                EeboFaissIndex.wipe_faiss_index(path)
        indices = build_indices(stream)

    elif not existing_years:
        logger.info("[faiss-build] no existing per-year indices found - building fresh")
        indices = build_indices(stream)

    else:
        logger.info(f"[faiss-build] incremental mode - loading {len(existing_years)} existing years")
        indices: dict[int, dict[str, EeboFaissIndex]] = {}
        already_indexed: set[int] = set()

        for year in existing_years:
            indices[year] = {}
            year_id_sets = {}

            for scale, path in faiss_index_paths(masked, year=year).items():
                idx = EeboFaissIndex.load(path)
                indices[year][scale] = idx
                year_id_sets[scale] = idx.ids()

            # Per-year sync check: all three scales for THIS year must hold
            # identical id sets, since they're populated together batch by
            # batch from the same year-slice of the stream.
            medium_ids = year_id_sets["medium"]
            for scale, ids in year_id_sets.items():
                if ids != medium_ids:
                    raise RuntimeError(
                        f"FAISS indices out of sync for year {year}: "
                        f"'{scale}' has {len(ids)} ids, 'medium' has "
                        f"{len(medium_ids)}. Rebuild with --clear."
                    )

            already_indexed |= medium_ids

        logger.info(f"[faiss-build] existing indices ntotal={len(already_indexed)} "
                    f"across {len(existing_years)} years")
        indices = build_indices(stream, indices=indices, already_indexed=already_indexed)

    persist_indicies(indices, masked)


def main():
    args = parse_args()

    zarr_path = (
        MASKED_ZARR_PATH
        if args.mask
        else ZARR_PATH
    )

    stream = ZarrEventStream(str(zarr_path))

    result = service(
        stream=stream,
        masked=args.mask,
        clear=args.clear,
    )

    logger.info( f"[faiss-build] complete: {result['summary']}" )


if __name__ == "__main__":
    main()
