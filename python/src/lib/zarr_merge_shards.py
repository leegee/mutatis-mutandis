#!/usr/bin/env python
"""
lib/zarr_merge_shards.py - Merge sharded Tier 1 Zarr stores into one final store.

Usage:
    python merge_shards.py --base-path /path/to/tier1.zarr --num-shards 3
"""

import argparse
from pathlib import Path

from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.corpus_logging import logger

CHUNK = 20_000  # rows per append_events call


def merge_shard(shard_path: Path, target: ZarrEmbeddingObservationStore):
    dim = target.embedding_dim() or target.dim
    src = ZarrEmbeddingObservationStore(path=str(shard_path), dim=dim)

    n = src.n_events
    if n == 0:
        logger.info("Shard %s is empty, skipping", shard_path)
        return 0

    logger.info("Merging %s: %d events", shard_path, n)

    existing_event_ids = target.get_event_ids()

    written = 0
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)

        event_id = src.event_id[start:end]

        # Skip rows whose event_id is already in the target store
        # (defensive — guards against re-running a merge after a partial failure)
        keep_mask = [eid not in existing_event_ids for eid in event_id]
        if not any(keep_mask):
            continue

        def sel(ds):
            arr = ds[start:end]
            return arr[keep_mask] if not all(keep_mask) else arr

        target.append_events(
            event_id          = sel(src.event_id),
            concept_id         = sel(src.concept_id),
            emb_local          = sel(src.emb_local),
            emb_medium         = sel(src.emb_medium),
            emb_broad          = sel(src.emb_broad),
            vector_id          = sel(src.vector_id),
            corpus             = sel(src.corpus),
            doc_id             = sel(src.doc_id),
            pub_year           = sel(src.pub_year),
            token_idx          = sel(src.token_idx),
            token              = sel(src.token),
            window_id          = sel(src.window_id),
            window_token_pos   = sel(src.window_token_pos),
        )
        written += sum(keep_mask)

    logger.info("Merged %d new events from %s (%d already present, skipped)",
                written, shard_path, n - written)
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-path", required=True, help="Final merged store path (no _shardN suffix)")
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument("--dim", type=int, default=768, help="Embedding dim (MacBERTh hidden size)")
    args = p.parse_args()

    base_path = Path(args.base_path)
    target = ZarrEmbeddingObservationStore(path=str(base_path), dim=args.dim)

    total_written = 0
    for i in range(args.num_shards):
        shard_path = base_path.parent / f"{base_path.name}_shard{i}"
        if not shard_path.exists():
            logger.warning("Shard path %s does not exist, skipping", shard_path)
            continue
        total_written += merge_shard(shard_path, target)

    logger.info("Merge complete: %d total new events, %d events in final store",
                total_written, target.n_events)


if __name__ == "__main__":
    main()
