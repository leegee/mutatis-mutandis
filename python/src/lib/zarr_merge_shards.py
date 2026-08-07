#!/usr/bin/env python
"""
lib/zarr_merge_shards.py - Merge sharded Tier 1 Zarr stores into one final store.

After a sharded run of tier1_0_corpus2zarr.py (orchestrated by
run_tier1_0_sharded.py), each worker writes to a path of the form
``{base}_shard{N}``. This module concatenates those shard stores into
the canonical Tier 1 observation store at ``base``, skipping any
event_id that is already present (defensive against partial re-merges).

Usage
-----

    python -m lib.zarr_merge_shards --base-path /path/to/tier1.zarr --num-shards 3

Or, more commonly, invoked automatically at the end of
run_tier1_0_sharded.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from lib.corpus_logging import logger
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore

CHUNK = 20_000  # rows per append_events call


def merge_shard(shard_path: Path, target: ZarrEmbeddingObservationStore) -> int:
    dim = target.embedding_dim() or target.dim

    src = ZarrEmbeddingObservationStore(
        path=str(shard_path),
        dim=dim,
    )

    if src.embedding_dim() != dim:
        raise ValueError(
            f"[merge] Embedding dimension mismatch: "
            f"{shard_path}={src.embedding_dim()} target={dim}"
        )

    n = src.n_events
    if n == 0:
        logger.info("[merge] Shard %s is empty, skipping", shard_path)
        return 0

    logger.info("[merge] Merging %s: %d events", shard_path, n)

    existing_event_ids = set(target.get_event_ids())

    written = 0

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)

        event_id = src.event_id[start:end]

        keep_mask = np.array(
            [eid not in existing_event_ids for eid in event_id],
            dtype=bool,
        )

        if not keep_mask.any():
            continue

        def sel(ds):
            arr = ds[start:end]
            return arr[keep_mask] if not keep_mask.all() else arr

        target.append_events(
            event_id            = sel(src.event_id),
            concept_id          = sel(src.concept_id),
            emb_local           = sel(src.emb_local),
            emb_medium          = sel(src.emb_medium),
            emb_broad           = sel(src.emb_broad),
            vector_id           = sel(src.vector_id),
            doc_id              = sel(src.doc_id),
            corpus              = sel(src.corpus),
            pub_year            = sel(src.pub_year),
            token_idx           = sel(src.token_idx),
            token               = sel(src.token),
            window_id           = sel(src.window_id),
            window_token_pos    = sel(src.window_token_pos),
        )

        new_ids = event_id[keep_mask]
        existing_event_ids.update(new_ids)

        written += int(keep_mask.sum())

    logger.info(
        "[merge] Merged %d new events from %s (%d already present, skipped)",
        written,
        shard_path,
        n - written,
    )

    return written


def main():
    p = argparse.ArgumentParser(
        description="Merge sharded Tier 1 Zarr stores into one final store"
    )
    p.add_argument(
        "--base-path",
        required=True,
        help="Final merged store path (no _shardN suffix)",
    )
    p.add_argument("--num-shards", type=int, required=True)
    p.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Embedding dim (MacBERTh hidden size)",
    )
    args = p.parse_args()

    base_path = Path(args.base_path)
    target = ZarrEmbeddingObservationStore(path=str(base_path), dim=args.dim)

    total_written = 0
    for i in range(args.num_shards):
        shard_path = base_path.parent / f"{base_path.name}_shard{i}"
        if not shard_path.exists():
            logger.warning("[merge] Shard path %s does not exist, skipping", shard_path)
            continue

        complete_marker = shard_path / "_COMPLETE"
        if not complete_marker.exists():
            raise RuntimeError(
                f"[merge] Refusing incomplete shard: {shard_path}"
            )
        total_written += merge_shard(shard_path, target)

    logger.info(
        "[merge] Merge complete: %d total new events, %d events in final store",
        total_written, target.n_events,
    )


if __name__ == "__main__":
    main()
