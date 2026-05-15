#!/usr/bin/env python
"""
tier1_test_repro.py

Checks embedding stability under different batching regimes.

Invariant tested:
    embeddings(doc) should be stable under changes to:
    - EMBED_BATCH_SIZE
    - forward chunking order

This isolates:
    - floating point accumulation effects
    - batching artefacts
"""

import tempfile
import shutil
import numpy as np

from dataclasses import dataclass

from lib.eebo_db import get_connection
from lib.macberth import load_macberth
from lib.vector_store_zarr import ZarrVectorStore


# ------------------------------------------------------------
# Config injection layer
# ------------------------------------------------------------

@dataclass
class RunConfig:
    zarr_root: str
    embed_batch_size: int
    window_size: int = 512


# ------------------------------------------------------------
# Core pipeline import (you already have this logic)
# ------------------------------------------------------------
# We assume you refactor process_doc to accept config:

from tier1_corpus2zarr import process_doc  # same function, but parameterised


# ------------------------------------------------------------
# single-document extraction
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
# run one configuration
# ------------------------------------------------------------

def run(config: RunConfig, tokens, vids, tokenizer, model, device):
    import tier1_corpus2zarr as mod

    # monkey-patch config knobs TODO: refactor properly later
    mod.ZARR_ROOT = config.zarr_root
    mod.EMBED_BATCH_SIZE = config.embed_batch_size
    mod.WINDOW_SIZE = config.window_size

    vecs, ids = process_doc(tokens, vids, tokenizer, model, device)

    return vecs, ids


# ------------------------------------------------------------
# comparison metric
# ------------------------------------------------------------

def compare(a, b):
    assert len(a) == len(b)

    cos = np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )

    return {
        "mean_cosine": float(np.mean(cos)),
        "min_cosine": float(np.min(cos)),
        "p01": float(np.percentile(cos, 1)),
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    conn = get_connection()
    mac = load_macberth()

    # choose ONE document (important for signal clarity)
    slice_range = (1641, 1641)
    tokens, vids = extract_doc(conn, slice_range)

    conn.close()

    tmp1 = tempfile.mkdtemp()
    tmp2 = tempfile.mkdtemp()

    config_a = RunConfig(tmp1, embed_batch_size=8)
    config_b = RunConfig(tmp2, embed_batch_size=64)

    vecs_a, ids_a = run(config_a, tokens, vids, mac.tokenizer, mac.model, mac.device)
    vecs_b, ids_b = run(config_b, tokens, vids, mac.tokenizer, mac.model, mac.device)

    assert np.array_equal(ids_a, ids_b), "ID mismatch between runs"

    stats = compare(vecs_a, vecs_b)

    print("\n=== REPRODUCIBILITY REPORT ===")
    print(stats)

    # cleanup
    shutil.rmtree(tmp1)
    shutil.rmtree(tmp2)


if __name__ == "__main__":
    main()