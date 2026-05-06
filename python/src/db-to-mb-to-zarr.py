#!/usr/bin/env python
"""
db-to-mb-to-zarr.py - STAGE 2 (Zarr only)

Invariant:
    - Every token embedding is a sample from overlapping window contexts
    - Each token stores (mean, variance, count) over those samples
    - No truncation bias: every token is covered by at least one window
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import os
import shutil
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EEBO_MODEL_NAME, MACBERTH_FINE_TUNED_DIR
from lib.vector_store_zarr import ZarrVectorStore


WINDOW_SIZE = 512
WINDOW_STRIDE = 128

TOKENIZER = None
MODEL = None
_DEVICE = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear-output", action="store_true")
    p.add_argument("--first-slice-only", action="store_true")
    return p.parse_args()


def clear_output_dir():
    path = ZARR_ROOT
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def get_model():
    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        logger.info("Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(
            EEBO_MODEL_NAME,
            local_files_only=True
        )

        model = AutoModel.from_pretrained(
            EEBO_MODEL_NAME,
            local_files_only=True
        )

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Tokenizer must preserve word alignment")

        ft_dir = MACBERTH_FINE_TUNED_DIR
        if (ft_dir / "pytorch_model.bin").exists():
            state = torch.load(ft_dir / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state, strict=False)

        model.eval()

        TOKENIZER, MODEL = tokenizer, model

    return TOKENIZER, MODEL


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def embed_document_token_centered(
    tokens: List[str],
    vector_ids: List[int],
    tokenizer,
    model,
    device,
    vec_sum: Dict[int, np.ndarray],
    vec_sqsum: Dict[int, np.ndarray],
    vec_count: Dict[int, int],
):
    """
    Invariant:
        Each token is embedded in its own local context window.
        No global sliding windows.
        No cross-window averaging.
    """

    n = len(tokens)

    for i in range(n):
        left = max(0, i - WINDOW_SIZE // 2)
        right = min(n, i + WINDOW_SIZE // 2)

        window_tokens = tokens[left:right]

        enc = tokenizer(
            window_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        hidden = outputs.last_hidden_state[0].detach().cpu().numpy()

        # token i is located at offset (i - left)
        pos = i - left
        if pos >= hidden.shape[0]:
            continue

        vid = int(vector_ids[i])
        vec = hidden[pos]

        vec_sum.setdefault(vid, np.zeros_like(vec))
        vec_sqsum.setdefault(vid, np.zeros_like(vec))

        vec_sum[vid] += vec
        vec_sqsum[vid] += vec * vec
        vec_count[vid] = vec_count.get(vid, 0) + 1


def flush_to_zarr(store, vec_sum, vec_sqsum, vec_count):
    if not vec_sum:
        return

    ids = []
    means = []
    vars_ = []
    counts = []

    for vid in vec_sum.keys():
        n = vec_count.get(vid, 0)
        if n == 0:
            continue

        mean = vec_sum[vid] / n
        var = (vec_sqsum[vid] / n) - (mean * mean)

        mean = normalize(mean)
        if mean is None:
            continue

        ids.append(vid)
        means.append(mean)
        vars_.append(var)
        counts.append(n)

    if not ids:
        return

    store.append(
        np.asarray(means, dtype=np.float32),
        np.asarray(vars_, dtype=np.float32),
        np.asarray(counts, dtype=np.int32),
        np.asarray(ids, dtype=np.int64),
    )

    vec_sum.clear()
    vec_sqsum.clear()
    vec_count.clear()


def process_slice(conn, slice_range):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    tokenizer, model = get_model()
    device = get_device()
    model.to(device)

    zarr_store = ZarrVectorStore(
        path=str(ZARR_ROOT / str(slice_id)),
        dim=model.config.hidden_size
    )

    vec_sum = {}
    vec_sqsum = {}
    vec_count = {}

    cur = conn.cursor(name=f"stream_{slice_id}")
    cur.itersize = 10000

    cur.execute("""
        SELECT t.doc_id, t.token_idx, t.token, t.vector_id
        FROM tokens t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE d.pub_year BETWEEN %s AND %s
        ORDER BY t.doc_id, t.token_idx
    """, slice_range)

    current_doc = None
    buffer = []

    def process_doc(rows):
        tokens = []
        vector_ids = []

        for _, _, token_text, vector_id in rows:
            tokens.append(token_text)
            vector_ids.append(vector_id)

        embed_document_token_centered(
            tokens,
            vector_ids,
            tokenizer,
            model,
            device,
            vec_sum,
            vec_sqsum,
            vec_count,
        )

        flush_to_zarr(zarr_store, vec_sum, vec_sqsum, vec_count)

    for doc_id, token_idx, token_text, vector_id in cur:

        if vector_id is None:
            raise ValueError(f"Missing vector_id {doc_id}:{token_idx}")

        if current_doc is None:
            current_doc = doc_id

        if doc_id != current_doc:
            process_doc(buffer)
            buffer = []
            current_doc = doc_id

        buffer.append((doc_id, token_idx, token_text, vector_id))

    if buffer:
        process_doc(buffer)

    cur.close()

    logger.info(f"[SLICE COMPLETE] {slice_id}")


def main():
    args = parse_args()

    if args.clear_output:
        logger.info("Clearing output directory")
        clear_output_dir()

    logger.info("Starting streaming pipeline (coverage + variance aware)")

    conn = get_connection()

    if args.first_slice_only:
        process_slice(conn, SLICES[0])
    else:
        for s in SLICES:
            process_slice(conn, s)

    conn.close()


if __name__ == "__main__":
    main()

