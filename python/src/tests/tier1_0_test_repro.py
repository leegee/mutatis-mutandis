#!/usr/bin/env python
"""
tier1_test_integrity.py

Embedding integrity test for EEBO pipeline.

Tests:
1. Batch invariance (EMBED_BATCH_SIZE)
2. Window invariance (WINDOW_SIZE)
3. Identity invariance (vector_id alignment)

We validate embedding stability across batching regimes.
"""

import numpy as np
from dataclasses import dataclass

from lib.corpus_db import get_connection
from lib.macberth import load_macberth
from lib.eebo_logging import logger

from tier1_corpus2zarr import process_doc

DEBUG = True


@dataclass
class RunConfig:
    embed_batch_size: int
    window_size: int = 512


def extract_doc(conn, slice_range):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT token, vector_id
            FROM pamphlet_tokens t
            JOIN pamphlet_corpus d ON d.doc_id = t.doc_id
            WHERE d.pub_year BETWEEN %s AND %s
            ORDER BY t.doc_id, t.token_idx
        """, slice_range)

        rows = list(cur)

    tokens = [r[0] for r in rows]
    vids = [r[1] for r in rows]

    if DEBUG:
        logger.info("DB token types=%s", [type(t) for t in tokens[:10]])
        logger.info("DB sizes tokens=%d vids=%d", len(tokens), len(vids))

    return tokens, vids


def run(config, doc_id, tokens, vids, tokenizer, model, device):
    import tier1_corpus2zarr as mod

    mod.EMBED_BATCH_SIZE = config.embed_batch_size
    mod.WINDOW_SIZE = config.window_size

    result = mod.process_doc(
        doc_id,
        tokens,
        vids,
        tokenizer,
        model,
        device,
    )

    # process_doc returns (vecs, ids, token_idxs)
    vecs, ids, _ = result

    if DEBUG:
        logger.info(
            "RUN batch=%d window=%d vecs_shape=%s",
            config.embed_batch_size,
            config.window_size,
            vecs.shape,
        )

    return vecs, ids


# Metrics
def cosine_stability(a, b):
    assert len(a) == len(b)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.clip(denom, 1e-12, None)

    return np.sum(a * b, axis=1) / denom


def report(cos):
    return {
        "mean_cosine": float(np.mean(cos)),
        "std_cosine": float(np.std(cos)),
        "p01": float(np.percentile(cos, 1)),
        "p50": float(np.percentile(cos, 50)),
        "p99": float(np.percentile(cos, 99)),
        "min_cosine": float(np.min(cos)),
        "unstable_fraction(<0.98)": float(np.mean(cos < 0.98)),
        "stable": bool(np.mean(cos) > 0.995),
    }


def main():
    conn = get_connection()
    mac = load_macberth()

    slice_range = (1641, 1641)
    tokens, vids = extract_doc(conn, slice_range)
    conn.close()

    doc_id = slice_range[0]

    configs = [
        RunConfig(embed_batch_size=8),
        RunConfig(embed_batch_size=16),
        RunConfig(embed_batch_size=32),
        RunConfig(embed_batch_size=64),
    ]

    # --------------------------------------------------------
    # Reference run
    # --------------------------------------------------------

    vecs_ref, ids_ref = run(
        configs[0],
        doc_id,
        tokens,
        vids,
        mac.tokenizer,
        mac.model,
        mac.device,
    )

    assert np.array_equal(ids_ref, vids), "vector_id misalignment in reference run"

    if DEBUG:
        logger.info("REFERENCE vecs shape=%s", vecs_ref.shape)

    # --------------------------------------------------------
    # Stability comparisons
    # --------------------------------------------------------

    print("\n=== EMBEDDING INTEGRITY REPORT ===")

    for cfg in configs[1:]:

        vecs, ids = run(
            cfg,
            doc_id,
            tokens,
            vids,
            mac.tokenizer,
            mac.model,
            mac.device,
        )

        assert np.array_equal(ids, vids), "vector_id mismatch across runs"

        cos = cosine_stability(vecs_ref, vecs)
        stats = report(cos)

        print(f"\nBatch size = {cfg.embed_batch_size}")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        if not stats["stable"]:
            print("  WARNING: instability detected")


if __name__ == "__main__":
    main()
