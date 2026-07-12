from __future__ import annotations

import argparse
import numpy as np

from lib.eebo_config import ZARR_PATH, MASKED_ZARR_PATH, FAISS_TIER1_INDEX, FAISS_TIER1_INDEX_MASKED
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_stream import ZarrEventStream


BATCH_SIZE = 8192


def build_index(
    stream: ZarrEventStream,
    index: EeboFaissIndex | None = None,
    already_indexed: set[int] | None = None,
) -> EeboFaissIndex:
    """
    Build or incrementally update FAISS index using multi-window ensemble embeddings.
    """
    total = 0
    skipped = 0
    incremental = already_indexed is not None

    logger.info("[faiss-build] streaming Tier1 multi-scale embeddings")

    for emb_local, emb_medium, emb_broad, obs_ids in stream.iter_multi_scale_embeddings(
        batch_size=BATCH_SIZE
    ):
        if len(obs_ids) == 0:
            continue

        ensemble = (
            0.25 * emb_local +
            0.50 * emb_medium +
            0.25 * emb_broad
        )

        if index is None:
            dim = ensemble.shape[1]
            index = EeboFaissIndex(dim=dim, exact=True)

        if incremental:
            new_mask = np.array([int(i) not in already_indexed for i in obs_ids])
            if not new_mask.any():
                skipped += len(obs_ids)
                continue

            ensemble = ensemble[new_mask]
            obs_ids = obs_ids[new_mask]
            skipped += (~new_mask).sum()

        try:
            index.add(ensemble, obs_ids)
            total += len(obs_ids)
            if total % 100_000 == 0:
                logger.info(f"[faiss-build] indexed {total:,} events so far...")
        except Exception as e:
            logger.error(f"[faiss-build] add failed: {e}", exc_info=True)
            raise

    if index is None:
        raise RuntimeError("No embeddings found in Tier1 observation store")

    logger.info(f"[faiss-build] finished - added={total:,} skipped={skipped:,}")
    return index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear", action="store_true",
                   help="Wipe existing FAISS index and rebuild from scratch")
    p.add_argument("--no-mask", action="store_true",
                   help="Build index from the unmasked Tier 1 store instead of the masked one")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_mask:
        zarr_path = ZARR_PATH
        faiss_index_path = FAISS_TIER1_INDEX
    else:
        zarr_path = MASKED_ZARR_PATH
        faiss_index_path = FAISS_TIER1_INDEX_MASKED

    logger.info(f"[faiss-build] mode={'unmasked' if args.no_mask else 'masked'} "
                f"zarr={zarr_path} index={faiss_index_path}")

    stream = ZarrEventStream(str(zarr_path))

    if args.clear or not faiss_index_path.is_file():
        if args.clear:
            logger.info("[faiss-build] clearing existing FAISS index")
            EeboFaissIndex.wipe_faiss_index(faiss_index_path.parent)

        logger.info("[faiss-build] building FAISS observation index from scratch")
        index = build_index(stream)
    else:
        logger.info("[faiss-build] incremental mode — loading existing index")
        index = EeboFaissIndex.load(faiss_index_path)
        already_indexed = index.ids()
        logger.info(f"[faiss-build] existing index ntotal={len(already_indexed)}")
        index = build_index(stream, index=index, already_indexed=already_indexed)

    faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(faiss_index_path)
    logger.info(f"[faiss-build] done -> {faiss_index_path}")

if __name__ == "__main__":
    main()

