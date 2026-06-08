#!/usr/bin/env python
"""
tier1_5_build_faiss_index.py

Shard-aware FAISS index construction over Tier 1 contextual observation stores.

Architecture
------------

Tier 1 is now a sharded store:

    ZARR_ROOT/<corpus>/<period>/<model>/<strategy>/events/*

Each shard gets its own FAISS index at the mirrored path:

    FAISS_ROOT/<corpus>/<period>/<model>/<strategy>/index.faiss

This means:
    - indexes are independently buildable and loadable
    - a period can be re-indexed without touching others
    - the search layer can query one shard, a corpus, a strategy,
      or all shards by composing shard lists from ShardResolver

Build modes
-----------

--all           Build/update indexes for all existing shards (default)
--corpus EEBO   Restrict to one corpus
--strategy      Restrict to one window strategy tag (e.g. sliding_512_256)
--clear         Wipe and rebuild from scratch for the selected shards
--shard PATH    Build/update a single explicit shard path

Incremental updates
-------------------

For each shard, get_indexed_ids() reads the existing FAISS id_map to find
already-indexed event_ids.  Only new events are added.

Scale note: at full EEBO+ECCO scale, materialising all indexed IDs into a
Python set per shard is manageable because each shard covers a 50-year
window of one corpus under one strategy — typically hundreds of thousands
of events, not hundreds of millions.  The global cross-shard case is
avoided by the per-shard architecture.

Key invariants
--------------
1. Tier 1 observation stores are the sole source of truth for embeddings.
2. FAISS stores only L2-normalised vectors and stable observation IDs.
3. vector_id is lexical identity — NOT embedding identity.
4. No full-corpus materialisation occurs during index construction.
5. Each shard's FAISS index is independent and self-contained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

from lib.eebo_config import ZARR_ROOT, FAISS_ROOT
from lib.eebo_logging import logger
from lib.eebo_faiss import EeboFaissIndex
from lib.shard_resolver import ShardResolver
from lib.window_strategy import WindowStrategy, WINDOW_STRATEGIES
from lib.zarr_event_stream import ZarrEventStream


BATCH_SIZE = 8192


# ---------------------------------------------------------------------------
# FAISS path resolution — mirrors the Zarr shard layout
# ---------------------------------------------------------------------------

def faiss_path_for_shard(shard_path: Path) -> Path:
    """
    Derive the FAISS index path for a Zarr shard path by replacing
    ZARR_ROOT with FAISS_ROOT and appending index.faiss.

        ZARR_ROOT/EEBO/1600-1649/MacBERTh/sliding_512_256
        ->
        FAISS_ROOT/EEBO/1600-1649/MacBERTh/sliding_512_256/index.faiss
    """
    rel  = shard_path.relative_to(ZARR_ROOT)
    path = FAISS_ROOT / rel / "index.faiss"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Indexed ID retrieval
# ---------------------------------------------------------------------------

def get_indexed_ids(index: EeboFaissIndex) -> set[int]:
    """
    Return the set of event_ids already present in a FAISS index.

    This is called once per shard, not globally, so the set stays
    manageable even at full corpus scale.
    """
    return set(faiss.vector_to_array(index._index.id_map).tolist())


# ---------------------------------------------------------------------------
# Per-shard build / update
# ---------------------------------------------------------------------------

def build_shard_index(
    shard_path:      Path,
    index:           EeboFaissIndex | None = None,
    already_indexed: set[int]       | None = None,
) -> EeboFaissIndex:
    """
    Build or incrementally update a FAISS index for a single shard.

    Parameters
    ----------
    shard_path:
        Path to the Zarr shard directory.
    index:
        Existing EeboFaissIndex to update.  None = build from scratch.
    already_indexed:
        Set of event_ids already in the index.  If None, all events
        in the shard are indexed.

    Returns
    -------
    EeboFaissIndex with all events from this shard indexed.
    """
    stream      = ZarrEventStream(shard_paths=[shard_path])
    incremental = already_indexed is not None
    total       = 0
    skipped     = 0

    logger.info(f"[faiss-build] shard={shard_path.relative_to(ZARR_ROOT)}")

    for vecs, obs_ids in stream.iter_embeddings(batch_size=BATCH_SIZE):
        if vecs is None or len(vecs) == 0:
            continue

        if index is None:
            dim   = vecs.shape[1]
            index = EeboFaissIndex(dim=dim, exact=True)
            logger.info(f"[faiss-build] created index dim={dim}")

        if incremental:
            new_mask = np.array(
                [int(i) not in already_indexed for i in obs_ids],
                dtype=bool,
            )
            if not new_mask.any():
                skipped += len(obs_ids)
                continue
            vecs    = vecs[new_mask]
            obs_ids = obs_ids[new_mask]
            skipped += int((~new_mask).sum())

        try:
            index.add(vecs, obs_ids)
            total += len(obs_ids)
        except Exception as exc:
            logger.error(
                f"[faiss-build] add failed for shard "
                f"{shard_path.relative_to(ZARR_ROOT)}: {exc}",
                exc_info=True,
            )
            raise

    if index is None:
        raise RuntimeError(
            f"No embeddings found in shard: {shard_path}"
        )

    logger.info(
        f"[faiss-build] shard done — added={total} skipped={skipped} "
        f"ntotal={index.ntotal}"
    )
    return index


# ---------------------------------------------------------------------------
# Multi-shard orchestration
# ---------------------------------------------------------------------------

def build_all(
    shard_paths: list[Path],
    clear:       bool = False,
) -> None:
    """
    Build or update FAISS indexes for a list of shard paths.
    Each shard is handled independently and saved immediately after
    completion — a crash mid-run leaves completed shards intact.
    """
    if not shard_paths:
        logger.warning("[faiss-build] no shards found — nothing to index")
        return

    logger.info(f"[faiss-build] processing {len(shard_paths)} shard(s)")

    for shard_path in shard_paths:
        index_path = faiss_path_for_shard(shard_path)
        rel        = shard_path.relative_to(ZARR_ROOT)

        if clear or not index_path.is_file():
            if clear and index_path.is_file():
                logger.info(f"[faiss-build] clearing {rel}")
                EeboFaissIndex.wipe_faiss_index(index_path)
            else:
                logger.info(f"[faiss-build] no existing index for {rel} — building")

            index = build_shard_index(shard_path)

        else:
            logger.info(f"[faiss-build] incremental update for {rel}")
            index           = EeboFaissIndex.load(index_path)
            already_indexed = get_indexed_ids(index)
            logger.info(
                f"[faiss-build] existing ntotal={len(already_indexed)}"
            )
            index = build_shard_index(
                shard_path,
                index           = index,
                already_indexed = already_indexed,
            )

        index.save(index_path)
        logger.info(f"[faiss-build] saved -> {index_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build or update per-shard FAISS indexes for Tier 1."
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Wipe and rebuild selected indexes from scratch",
    )
    p.add_argument(
        "--corpus",
        type=str,
        default=None,
        metavar="CORPUS_ID",
        help="Restrict to one corpus (e.g. EEBO, ECCO)",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default=None,
        metavar="STRATEGY_TAG",
        help=(
            "Restrict to one window strategy tag "
            "(e.g. sliding_512_256, doc, sentence)"
        ),
    )
    p.add_argument(
        "--shard",
        type=str,
        default=None,
        metavar="PATH",
        help="Build/update a single explicit shard path",
    )
    p.add_argument(
        "--model",
        type=str,
        default="MacBERTh",
        help="Model name used in shard path resolution (default: MacBERTh)",
    )
    return p.parse_args()


def main():
    args     = parse_args()
    resolver = ShardResolver(model_name=args.model)

    if args.shard:
        shard_paths = [Path(args.shard)]
    else:
        # Resolve strategy filter if supplied
        strategy_filter: WindowStrategy | None = None
        if args.strategy:
            # Match by tag against the registered strategies
            matches = [s for s in WINDOW_STRATEGIES if s.tag == args.strategy]
            if not matches:
                raise ValueError(
                    f"Unknown strategy tag {args.strategy!r}. "
                    f"Known: {[s.tag for s in WINDOW_STRATEGIES]}"
                )
            strategy_filter = matches[0]

        shard_paths = resolver.all_shards(
            corpus_id = args.corpus,
            strategy  = strategy_filter,
        )

    build_all(shard_paths, clear=args.clear)


if __name__ == "__main__":
    main()
