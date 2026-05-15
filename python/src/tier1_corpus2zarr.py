#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Clean architecture:
-------------------
- Tokenise once per document into subword space
- Window over subword sequence only
- No re-tokenisation downstream
- No word/subword mixed indexing in runtime logic

Core invariant:
    ALL MODEL OPERATIONS ARE IN SUBWORD SPACE
    ALL AGGREGATION IS IN WORD SPACE (metadata only)
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


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Window:
    input_ids: List[int]
    attention_mask: List[int]
    word_ids: List[Optional[int]]
    token_offset: int


@dataclass
class PendingDoc:
    vec_sum: Dict[int, np.ndarray] = field(default_factory=dict)
    count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    vector_ids_by_word: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Sidecar index
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Tokenisation + windowing (SUBWORD ONLY)
# ---------------------------------------------------------------------

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


def make_windows(input_ids, attention_mask, word_ids):
    windows = []
    n = len(input_ids)

    start = 0
    while start < n:
        end = min(start + WINDOW_SIZE, n)

        windows.append(Window(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            word_ids=word_ids[start:end],
            token_offset=start,
        ))

        if end == n:
            break
        start += STRIDE

    return windows


# ---------------------------------------------------------------------
# Forward pass (NO TOKENIZER HERE)
# ---------------------------------------------------------------------

def forward_windows(windows, model, device):
    max_len = max(len(w.input_ids) for w in windows)

    def pad(seq, pad_id=0):
        return seq + [pad_id] * (max_len - len(seq))

    batch_input = torch.tensor(
        [pad(w.input_ids) for w in windows]
    ).to(device)

    batch_mask = torch.tensor(
        [pad(w.attention_mask, 0) for w in windows]
    ).to(device)

    with torch.no_grad():
        out = model(
            input_ids=batch_input,
            attention_mask=batch_mask,
            return_dict=True
        )

    hidden = out.last_hidden_state.cpu().numpy()

    return [(windows[i], hidden[i]) for i in range(len(windows))]


# ---------------------------------------------------------------------
# Accumulation (word-space aggregation only)
# ---------------------------------------------------------------------

def accumulate(pending: PendingDoc, window: Window, hidden: np.ndarray):
    for i, word_id in enumerate(window.word_ids):
        if word_id is None:
            continue

        vec = normalize(hidden[i])
        if vec is None:
            continue

        if word_id not in pending.vec_sum:
            pending.vec_sum[word_id] = np.zeros(vec.shape, dtype=np.float64)

        pending.vec_sum[word_id] += vec
        pending.count[word_id] += 1


def finalise(pending: PendingDoc):
    vecs, ids = [], []

    for word_id in sorted(pending.vec_sum):
        c = pending.count[word_id]
        if c == 0:
            continue

        vecs.append((pending.vec_sum[word_id] / c).astype(np.float32))
        ids.append(pending.vector_ids_by_word[word_id])

    return (
        np.array(vecs, dtype=np.float32),
        np.array(ids, dtype=np.int64),
    )


# ---------------------------------------------------------------------
# Per-document pipeline
# ---------------------------------------------------------------------

def process_doc(tokens, vector_ids, tokenizer, model, device):

    input_ids, attention_mask, word_ids = encode_doc(tokens, tokenizer)

    pending = PendingDoc()
    pending.vector_ids_by_word = {
        i: vector_ids[i] for i in range(len(vector_ids))
    }

    windows = make_windows(input_ids, attention_mask, word_ids)

    for chunk_start in range(0, len(windows), EMBED_BATCH_SIZE):
        chunk = windows[chunk_start:chunk_start + EMBED_BATCH_SIZE]
        results = forward_windows(chunk, model, device)

        for w, hidden in results:
            accumulate(pending, w, hidden)

    return finalise(pending)


# ---------------------------------------------------------------------
# Slice processing
# ---------------------------------------------------------------------

def process_slice(conn, slice_range, tokenizer, model, device, doc_index):

    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    store = ZarrVectorStore(
        path=str(ZARR_ROOT / "tier1" / slice_id),
        dim=model.config.hidden_size,
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

        vecs, ids = process_doc(buf_tokens, buf_vids, tokenizer, model, device)

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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

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
        process_slice(conn, s, mac.tokenizer, mac.model, mac.device, {})

    conn.close()


if __name__ == "__main__":
    main()
