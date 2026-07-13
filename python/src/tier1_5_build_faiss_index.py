from __future__ import annotations

import argparse
import numpy as np

from lib.eebo_config import ZARR_PATH, MASKED_ZARR_PATH,  faiss_index_paths
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream

BATCH_SIZE = 8192
SCALES = ("local", "medium", "broad")



def build_indices(
    stream: ZarrEventStream,
    indices: dict[str, EeboFaissIndex] | None = None,
    already_indexed: set[int] | None = None,
) -> dict[str, EeboFaissIndex]:
    """
    Build/update THREE FAISS indices (local, medium, broad) from a single
    streamed pass. Replaces the weighted-ensemble approach — each scale is
    indexed independently so fusion happens downstream at query time.
    """
    total = 0
    skipped = 0
    incremental = already_indexed is not None
    indices = indices or {}

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings")

    for emb_local, emb_medium, emb_broad, obs_ids in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE
    ):
        if len(obs_ids) == 0:
            continue

        per_scale = {"local": emb_local, "medium": emb_medium, "broad": emb_broad}

        for scale, emb in per_scale.items():
            if scale not in indices:
                indices[scale] = EeboFaissIndex(dim=emb.shape[1], exact=True)

        scale_obs_ids = obs_ids
        if incremental:
            new_mask = np.array([int(i) not in already_indexed for i in obs_ids])
            if not new_mask.any():
                skipped += len(obs_ids)
                continue
            scale_obs_ids = obs_ids[new_mask]
            skipped += (~new_mask).sum()
            per_scale = {s: e[new_mask] for s, e in per_scale.items()}

        try:
            # All three scales must stay in lockstep — same obs_ids added
            # to each index every batch — or the "sync check" on load below
            # will start failing.
            for scale, emb in per_scale.items():
                indices[scale].add(emb, scale_obs_ids)
            total += len(scale_obs_ids)
            if total % 100_000 == 0:
                logger.info(f"[faiss-build] indexed {total:,} events so far...")
        except Exception as e:
            logger.error(f"[faiss-build] add failed: {e}", exc_info=True)
            raise

    if not indices:
        raise RuntimeError("No embeddings found in Tier1 observation store")

    logger.info(f"[faiss-build] finished - added={total:,} skipped={skipped:,}")
    return indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true")
    p.add_argument("--no-mask", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    zarr_path = ZARR_PATH if args.no_mask else MASKED_ZARR_PATH
    paths = faiss_index_paths(masked=not args.no_mask)

    logger.info(f"[faiss-build] mode={'unmasked' if args.no_mask else 'masked'} "
                f"zarr={zarr_path} indices={paths}")

    stream = ZarrEventStream(str(zarr_path))
    any_missing = any(not p.is_file() for p in paths.values())

    if args.clear or any_missing:
        if args.clear:
            for scale, path in paths.items():
                logger.info(f"[faiss-build] clearing existing {scale} index")
                EeboFaissIndex.wipe_faiss_index(path)   # pass the file, not .parent — see earlier fix
        indices = build_indices(stream)
    else:
        logger.info("[faiss-build] incremental mode — loading existing indices")
        indices = {scale: EeboFaissIndex.load(path) for scale, path in paths.items()}

        # Sanity check: all three should hold identical id sets since
        # they're populated together, batch by batch, from the same stream.
        already_indexed = indices["medium"].ids()
        for scale, idx in indices.items():
            if idx.ids() != already_indexed:
                raise RuntimeError(
                    f"FAISS indices out of sync: '{scale}' has {len(idx.ids())} ids, "
                    f"'medium' has {len(already_indexed)}. Rebuild with --clear."
                )

        logger.info(f"[faiss-build] existing indices ntotal={len(already_indexed)}")
        indices = build_indices(stream, indices=indices, already_indexed=already_indexed)

    for scale, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        indices[scale].save(path)
        logger.info(f"[faiss-build] done -> {path}")


if __name__ == "__main__":
    main()
