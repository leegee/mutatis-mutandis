#!/usr/bin/env python
"""
tools/zarr_backfill_corpus.py

corpus is currently unused downstream and doc_id is globally unique at
this prototyping stage, so this backfills corpus with a constant
placeholder rather than joining against Postgres. Prototype-only —
if corpus becomes load-bearing (e.g. once doc_id uniqueness can no
longer be assumed), this backfill is invalid and the store needs a
real rebuild or a proper vector_id-based join instead.
"""

import numpy as np
import zarr

from lib.corpus_logging import logger
from lib.corpus_config import EVENTSTORE_T1_PATH

PLACEHOLDER = "unknown"


def main():
    g = zarr.open_group(str(EVENTSTORE_T1_PATH), mode="a")["events"]

    n = g["event_id"].shape[0]
    corpus_ds = g["corpus"]

    if corpus_ds.shape[0] == n:
        logger.info("[backfill] corpus already has %d rows -- nothing to do", n)
        return

    if corpus_ds.shape[0] != 0:
        raise RuntimeError(
            f"corpus has {corpus_ds.shape[0]} rows, expected 0 or {n}. "
            f"Not the clean case this script handles -- stop and check."
        )

    logger.info("[backfill] filling corpus with placeholder=%r for %d rows", PLACEHOLDER, n)
    corpus_ds.resize((n,))
    corpus_ds[:] = np.full(n, PLACEHOLDER, dtype="U32")

    logger.info("[backfill] done -- corpus now has %d rows", corpus_ds.shape[0])


if __name__ == "__main__":
    main()
