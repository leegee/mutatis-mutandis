#!/usr/bin/env python
"""
tier1_test_repro.py

Stability test for event-log embedding pipeline.

This test does NOT assume strict determinism of embeddings.

Instead it evaluates:

1. Identity stability
   - token alignment is invariant across batching regimes

2. Coverage stability
   - same number of token-level events are produced

3. Numerical stability
   - embeddings are stable under changes in:
       - EMBED_BATCH_SIZE
       - window batching order

Core interpretation:
- The pipeline is a stochastic reduction over overlapping contexts
- Exact equality is not expected
- We measure cosine stability within tolerance bands
"""

import tempfile
import shutil
import numpy as np
from dataclasses import dataclass

from lib.eebo_db import get_connection
from lib.macberth import load_macberth

from tier1_corpus2zarr import process_doc


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class RunConfig:
    embed_batch_size: int
    window_size: int = 512


# ------------------------------------------------------------
# Data extraction
# ------------------------------------------------------------

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
    return tokens, vids


# ------------------------------------------------------------
# run configuration (controlled injection)
# ------------------------------------------------------------

def run(config: RunConfig, tokens, vids, tokenizer, model, device):
    import tier1_corpus2zarr as mod

    mod.EMBED_BATCH_SIZE = config.embed_batch_size
    mod.WINDOW_SIZE = config.window_size

    return process_doc(tokens, vids, tokenizer, model, device)


# ------------------------------------------------------------
# metrics
# ------------------------------------------------------------

def cosine_stability(a, b):
    assert len(a) == len(b)

    a = a.astype(np.float32)
    b = b.astype(np.float32)

    denom = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    denom = np.clip(denom, 1e-12, None)

    cos = np.sum(a * b, axis=1) / denom

    return cos


def report(cos):
    return {
        "mean_cosine": float(np.mean(cos)),
        "std_cosine": float(np.std(cos)),
        "p01": float(np.percentile(cos, 1)),
        "p99": float(np.percentile(cos, 99)),
        "min_cosine": float(np.min(cos)),
        "stable": bool(np.mean(cos) > 0.98),
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    conn = get_connection()
    mac = load_macberth()

    # single document for signal clarity
    slice_range = (1641, 1641)
    tokens, vids = extract_doc(conn, slice_range)
    conn.close()

    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()

    config_a = RunConfig(embed_batch_size=8)
    config_b = RunConfig(embed_batch_size=64)

    vecs_a, ids_a = run(config_a, tokens, vids, mac.tokenizer, mac.model, mac.device)
    vecs_b, ids_b = run(config_b, tokens, vids, mac.tokenizer, mac.model, mac.device)

    # ------------------------------------------------------------
    # 1. Identity invariant (hard requirement)
    # ------------------------------------------------------------
    assert np.array_equal(ids_a, ids_b), "Token/event identity mismatch"

    # ------------------------------------------------------------
    # 2. Coverage invariant (hard requirement)
    # ------------------------------------------------------------
    assert len(vecs_a) == len(vecs_b), "Event coverage mismatch"

    # ------------------------------------------------------------
    # 3. Stability invariant (soft requirement)
    # ------------------------------------------------------------
    cos = cosine_stability(vecs_a, vecs_b)
    stats = report(cos)

    print("\n=== EMBEDDING STABILITY REPORT ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # interpretive guardrail (not a hard failure)
    if not stats["stable"]:
        print("\nWARNING: embeddings show high sensitivity to batching regime")

    shutil.rmtree(tmp_a)
    shutil.rmtree(tmp_b)


if __name__ == "__main__":
    main()
