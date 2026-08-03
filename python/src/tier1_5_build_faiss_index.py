from __future__ import annotations

import argparse
import re

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
) -> dict[int, dict[str, EeboFaissIndex]]:
    """
    Build/update per-year, per-scale FAISS indices from a single streamed
    pass. Each (year, scale) pair gets its own EeboFaissIndex, so downstream
    fusion can happen both across scales and within/across year ranges.
    """
    total = 0
    skipped = 0
    incremental = already_indexed is not None
    indices = indices or {}

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings by year")

    for emb_local, emb_medium, emb_broad, obs_ids, pub_years in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE
    ):
        if len(obs_ids) == 0:
            continue

        per_scale = {"local": emb_local, "medium": emb_medium, "broad": emb_broad}

        if incremental:
            # TODO Optimize
            new_mask = np.array([int(i) not in already_indexed for i in obs_ids])
            if not new_mask.any():
                skipped += len(obs_ids)
                continue
            skipped += (~new_mask).sum()
            obs_ids = obs_ids[new_mask]
            pub_years = pub_years[new_mask]
            per_scale = {s: e[new_mask] for s, e in per_scale.items()}

        # Split this batch by year, then add each year-slice to that year's
        # three scale indices in lockstep — same invariant as before, now
        # scoped per year instead of globally.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--mask", action="store_true")
    return p.parse_args()


def main():
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
        logger.info("[faiss-build] no existing per-year indices found — building fresh")
        indices = build_indices(stream)

    else:
        logger.info(f"[faiss-build] incremental mode — loading {len(existing_years)} existing years")
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

    for year, scale_indices in indices.items():
        paths = faiss_index_paths(masked, year=year)
        for scale, idx in scale_indices.items():
            path = paths[scale]
            path.parent.mkdir(parents=True, exist_ok=True)
            idx.save(path)
            logger.info(f"[faiss-build] done -> {path}")


if __name__ == "__main__":
    main()
