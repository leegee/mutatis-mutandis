#!/usr/bin/env python
"""
run_tier1_sharded.py - Orchestrate sharded Tier 1 Parquet embedding.

Launches --num-shards worker processes running
tier1_corpus2parquet.py, waits for all workers to finish, and leaves the
completed Parquet shard datasets in place.

Parquet shards are independent datasets and do not require a post-processing
merge. Downstream readers can treat the shard directories as one logical
dataset.

Usage:
    python run_tier1_sharded.py --num-shards 3 --threads-per-shard 2 --clear

A successful run produces, for example:

    <tier1-root>_shard0/
    <tier1-root>_shard1/
    <tier1-root>_shard2/

Each shard is marked complete by the worker only after its document stream
has finished successfully.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from lib.corpus_config import LOG_DIR
from lib.corpus_logging import logger
from tier1.tier1_corpus2parquet import clear_output_dir
from tier1.observation_store_api import resolve_store_path


T1_PATH = "src/tier1/tier1_corpus2parquet.py"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run sharded Tier 1 Parquet embedding"
    )

    p.add_argument(
        "--num-shards",
        type=int,
        default=3,
        help="Number of worker processes",
    )

    p.add_argument(
        "--threads-per-shard",
        type=int,
        default=1,
        help="CPU threads available to each worker",
    )

    p.add_argument(
        "--mask",
        action="store_true",
        help="Use masked target embeddings",
    )

    p.add_argument(
        "--clear",
        action="store_true",
        help="Wipe all shard stores before starting",
    )

    p.add_argument(
        "--report-every",
        type=int,
        default=100,
        help="Log progress every N completed documents per shard",
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the Tier 1 embedding batch size",
    )

    p.add_argument(
        "--backend",
        choices=["onnx", "pytorch"],
        default="onnx",
        help="Inference backend for embedding",
    )

    p.add_argument(
        "--onnx-provider",
        choices=["cpu", "dml"],
        default="cpu",
        help="ONNX Runtime provider",
    )

    p.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override the Tier 1 Parquet store root",
    )

    p.add_argument(
        "--parquet-min-rows",
        type=int,
        default=None,
        help="Flush Parquet after approximately this many buffered rows",
    )

    p.add_argument(
        "--parquet-min-bytes",
        type=int,
        default=None,
        help="Flush Parquet after approximately this many bytes",
    )

    return p.parse_args()


def shard_path(args, shard: int) -> Path:
    return resolve_store_path(
        store_backend="parquet",
        masked=args.mask,
        store=args.store,
        shard=shard,
        num_shards=args.num_shards,
    )


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be at least 1")

    if args.threads_per_shard < 1:
        raise SystemExit("--threads-per-shard must be at least 1")

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    shard_paths = [
        shard_path(args, shard)
        for shard in range(args.num_shards)
    ]

    logger.info(
        "Tier 1 Parquet: launching %d shards (%d threads each)",
        args.num_shards,
        args.threads_per_shard,
    )

    for shard, path in enumerate(shard_paths):
        logger.info(
            "Shard %d output: %s",
            shard,
            path,
        )

    if args.clear:
        for shard, path in enumerate(shard_paths):
            logger.info(
                "Clearing shard %d output: %s",
                shard,
                path,
            )
            clear_output_dir(path)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads_per_shard)
    env["MKL_NUM_THREADS"] = str(args.threads_per_shard)
    env["OPENBLAS_NUM_THREADS"] = str(args.threads_per_shard)
    env["TOKENIZERS_PARALLELISM"] = "false"

    processes = []

    for shard in range(args.num_shards):
        cmd = [
            sys.executable,
            T1_PATH,
            "--num-shards",
            str(args.num_shards),
            "--shard",
            str(shard),
            "--report-every",
            str(args.report_every),
            "--backend",
            args.backend,
            "--onnx-provider",
            args.onnx_provider,
            "--store-backend",
            "parquet",
        ]

        if args.mask:
            cmd.append("--mask")

        if args.clear:
            cmd.append("--clear")

        if args.batch_size is not None:
            cmd.extend(
                [
                    "--batch-size",
                    str(args.batch_size),
                ]
            )

        if args.store is not None:
            cmd.extend(
                [
                    "--store",
                    args.store,
                ]
            )

        if args.parquet_min_rows is not None:
            cmd.extend(
                [
                    "--parquet-min-rows",
                    str(args.parquet_min_rows),
                ]
            )

        if args.parquet_min_bytes is not None:
            cmd.extend(
                [
                    "--parquet-min-bytes",
                    str(args.parquet_min_bytes),
                ]
            )

        log_path = Path(LOG_DIR) / f"tier1_parquet_shard_{shard}.log"
        log_file = open(
            log_path,
            "w",
            encoding="utf-8",
        )

        logger.info(
            "Starting shard %d: %s (log: %s)",
            shard,
            " ".join(cmd),
            log_path,
        )

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        processes.append(
            (
                shard,
                proc,
                log_file,
                log_path,
            )
        )

    failed = []

    for shard, proc, log_file, log_path in processes:
        return_code = proc.wait()
        log_file.close()

        if return_code != 0:
            failed.append(shard)

            logger.error(
                "Shard %d failed (exit code %d) — see %s",
                shard,
                return_code,
                log_path,
            )
        else:
            logger.info(
                "Shard %d completed successfully",
                shard,
            )

    if failed:
        logger.error(
            "Tier 1 failed: shard(s) %s did not complete successfully. "
            "No combined dataset should be considered complete.",
            failed,
        )
        sys.exit(1)

    logger.info(
        "All %d Tier 1 Parquet shards completed successfully.",
        args.num_shards,
    )

    for shard, path in enumerate(shard_paths):
        logger.info(
            "Shard %d: %s",
            shard,
            path,
        )

    logger.info(
        "No merge required: Parquet shards remain independently readable "
        "as one logical dataset."
    )


if __name__ == "__main__":
    main()
