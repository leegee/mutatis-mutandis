#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Subword-native embedding pipeline (clean + memory optimised).

Core invariants:
- Tokenisation happens once per document
- Windowing operates only on subword sequence
- No word/subword mixed indexing in runtime logic
- No global window lists (streaming windows)

HIDDEN-STATE LAYER CHOICE IS IMPLICIT

We use `out.last_hidden_state` which means final transformer layer only.
This may not be optimal.

IDEALLY WE WANT:

- sentence-aware chunking
- paragraph-aware chunking
- discourse-aware chunking
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EMBED_BATCH_SIZE
from lib.vector_store_zarr import ZarrVectorStore
from lib.macberth import load_macberth, normalize


WINDOW_SIZE = 512
STRIDE = WINDOW_SIZE // 2

DOC_INDEX_PATH = ZARR_ROOT / "tier1" / "doc_index.json"



# Data structures


@dataclass
class Window:
    input_ids: List[int]
    attention_mask: List[int]
    word_ids: List[Optional[int]]
    token_offset: int


@dataclass
class PendingDoc:
    """
    Dense accumulator:
    - vec_sum: [num_tokens, hidden_dim] float64
    - count:   [num_tokens] int32
    - vector_ids: [num_tokens] int64
    """
    vec_sum: np.ndarray
    count: np.ndarray
    vector_ids: np.ndarray



# Sidecar index


def load_doc_index() -> Dict:
    if DOC_INDEX_PATH.exists():
        with open(DOC_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_doc_index(index: Dict) -> None:
    tmp = DOC_INDEX_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f)
    os.replace(tmp, DOC_INDEX_PATH)


def record_doc(index, slice_id, doc_id, start, end):
    index.setdefault(slice_id, {})[doc_id] = {"start": start, "end": end}
    save_doc_index(index)



# Tokenisation


def encode_doc(tokens, tokenizer):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=False,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"][0].tolist()
    attention_mask = enc["attention_mask"][0].tolist()
    word_ids = enc.word_ids()

    return input_ids, attention_mask, word_ids



# Streaming window generator


def iter_windows(input_ids, attention_mask, word_ids):
    n = len(input_ids)
    start = 0

    while start < n:
        end = min(start + WINDOW_SIZE, n)

        yield Window(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            word_ids=word_ids[start:end],
            token_offset=start,
        )

        if end == n:
            break
        start += STRIDE



# Forward pass
def forward_windows(windows, mac):
    results = []
    batch = []

    def flush(batch):
        if not batch:
            return

        max_len = len(batch[0].input_ids)

        def pad(seq, pad=0):
            return seq + [pad] * (max_len - len(seq))

        batch_input = torch.tensor(
            [pad(w.input_ids) for w in batch]
        ).to(device)

        batch_mask = torch.tensor(
            [pad(w.attention_mask, 0) for w in batch]
        ).to(device)

        with torch.inference_mode():
            out = mac.encode(
                input_ids=batch_input,
                attention_mask=batch_mask,
                return_dict=True,
            )

        hidden = out.last_hidden_state.detach().cpu().numpy()

        for i in range(len(batch)):
            results.append((batch[i], hidden[i]))

        batch.clear()

    for w in windows:
        batch.append(w)
        if len(batch) >= EMBED_BATCH_SIZE:
            flush(batch)

    flush(batch)

    return results


def accumulate(pending: PendingDoc, window: Window, hidden: np.ndarray):
    for i, word_id in enumerate(window.word_ids):
        if word_id is None:
            continue

        # ie remove vectors' magnitude ie frequnecy
        # Mayvbe keep an unnormalized version for comparison?
        vec = normalize(hidden[i])
        if vec is None:
            continue

        pending.vec_sum[word_id] += vec
        pending.count[word_id] += 1


def finalise(pending: PendingDoc):
    valid = pending.count > 0

    if not np.any(valid):
        return (
            np.empty((0, pending.vec_sum.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    vecs = (pending.vec_sum[valid] / pending.count[valid, None]).astype(np.float32)
    ids = pending.vector_ids[valid].astype(np.int64)

    return vecs, ids



# Per-document pipeline
def process_doc(tokens, vector_ids, mac):
    input_ids, attention_mask, word_ids = encode_doc(tokens, mac.tokenizer)

    n = len(tokens)
    dim = mac.model.config.hidden_size

    pending = PendingDoc(
        vec_sum=np.zeros((n, dim), dtype=np.float32),
        count=np.zeros(n, dtype=np.int32),
        vector_ids=np.array(vector_ids, dtype=np.int64),
    )

    window_iter = iter_windows(input_ids, attention_mask, word_ids)

    batch = []

    for w in window_iter:
        batch.append(w)

        if len(batch) >= EMBED_BATCH_SIZE:
            results = forward_windows(batch, mac)
            for win, hidden in results:
                accumulate(pending, win, hidden)
            batch.clear()

    if batch:
        results = forward_windows(batch, model, device)
        for win, hidden in results:
            accumulate(pending, win, hidden)

    return finalise(pending)


# Slice processor


def process_slice(conn, slice_range, mac, device, doc_index):

    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    store = ZarrVectorStore(
        path=str(ZARR_ROOT / "tier1" / slice_id),
        dim=mac.model.config.hidden_size,
    )

    cur = conn.cursor(name=f"tier1_{slice_id}")
    cur.itersize = 10000
    cur.execute("""
        SELECT t.doc_id, t.token_idx, t.vector_id, t.token
        FROM pamphlet_tokens t
        JOIN pamphlet_corpus d ON d.doc_id = t.doc_id
        WHERE d.pub_year BETWEEN %s AND %s
        ORDER BY t.doc_id, t.token_idx
    """, slice_range)

    current_doc = None
    buf_tokens = []
    buf_vids = []

    docs = 0

    def flush():
        nonlocal docs, current_doc

        if not buf_tokens:
            return

        vecs, ids = process_doc(buf_tokens, buf_vids, mac)

        if len(vecs) == 0:
            return

        assert len(set(ids.tolist())) == len(ids)

        start = len(store)
        store.append(vecs, ids)
        end = len(store)

        record_doc(doc_index, slice_id, current_doc, start, end)

        docs += 1
        if docs % 200 == 0:
            logger.info(f"[{slice_id}] {docs} docs")

    for doc_id, _, vid, token in cur:

        if current_doc is None:
            current_doc = doc_id

        if doc_id != current_doc:
            flush()
            buf_tokens.clear()
            buf_vids.clear()
            current_doc = doc_id

        buf_tokens.append(token)
        buf_vids.append(vid)

    flush()

    logger.info(f"[SLICE COMPLETE] {slice_id}")



# CLI (UNCHANGED)


def clear_output_dir():
    path = ZARR_ROOT / "tier1"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear-output", action="store_true")
    p.add_argument("--first-slice-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.clear_output:
        logger.info("Clearing Tier 1 output")
        clear_output_dir()

    conn = get_connection()
    mac = load_macberth()

    slices = SLICES[:1] if args.first_slice_only else SLICES

    for s in slices:
        process_slice(conn, s, mac, {})

    conn.close()


if __name__ == "__main__":
    main()
