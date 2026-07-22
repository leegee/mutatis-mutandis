#!/usr/bin/env python
"""
run_tier1_sharded.py - Orchestrate sharded Tier 1 embedding + merge.

Launches --num-shards worker processes running tier1_corpus2zarr.py,
waits for all to finish, then merges their output stores into one.

Usage:
    python run_tier1_sharded.py --num-shards 3 --threads-per-shard 2 --clear
"""

import argparse
import shutil
import subprocess
import sys
import os
from pathlib import Path

from lib.eebo_config import ZARR_PATH, MASKED_ZARR_PATH, LOG_DIR
from lib.eebo_logging import logger

T1_PATH = "src/tier1_0_corpus2zarr.py"

def clear_output_dir(zarr_path: Path):
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    zarr_path.mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--threads-per-shard", type=int, default=1)
    p.add_argument("--mask", action="store_true")
    p.add_argument("--clear", action="store_true",
                   help="Wipe shard stores AND the final merged target for a fully fresh run")
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--report-every", type=int, default=100)
    args = p.parse_args()

    base_path = MASKED_ZARR_PATH if args.mask else ZARR_PATH

    if args.clear:
        logger.info("Clearing final merge target: %s", base_path)
        clear_output_dir(base_path)
        # Note: each shard subprocess also receives --clear and wipes its own
        # _shardN path independently — see the cmd construction below.

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads_per_shard)
    env["MKL_NUM_THREADS"] = str(args.threads_per_shard)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_shard)
    env["TOKENIZERS_PARALLELISM"] = "false"

    logger.info("Launching %d shards (%d threads each)", args.num_shards, args.threads_per_shard)

    procs = []
    for i in range(args.num_shards):
        cmd = [
            sys.executable, T1_PATH,
            "--num-shards", str(args.num_shards),
            "--shard", str(i),
            "--report-every", str(args.report_every),
        ]
        if args.mask:
            cmd.append("--mask")
        if args.clear:
            cmd.append("--clear")

        log_path = Path(LOG_DIR / f"shard_{i}.log")
        log_file = open(log_path, "w")
        logger.info("Starting shard %d: %s (log: %s)", i, " ".join(cmd), log_path)

        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_file))

    failed = []
    for i, proc, log_file in procs:
        ret = proc.wait()
        log_file.close()
        if ret != 0:
            failed.append(i)
            logger.error("Shard %d failed (exit code %d) — see shard_%d.log", i, ret, i)
        else:
            logger.info("Shard %d completed successfully", i)

    if failed:
        logger.error("Aborting merge: shard(s) %s failed. Fix and rerun before merging.", failed)
        sys.exit(1)

    logger.info("All shards completed. Merging into %s", base_path)

    from lib.zarr_merge_shards import merge_shard
    from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore

    target = ZarrEmbeddingObservationStore(path=str(base_path), dim=args.dim)
    total_written = 0
    for i in range(args.num_shards):
        shard_path = base_path.parent / f"{base_path.name}_shard{i}"
        total_written += merge_shard(shard_path, target)

    logger.info("Merge complete: %d new events, %d total in store", total_written, target.n_events)


if __name__ == "__main__":
    main()
