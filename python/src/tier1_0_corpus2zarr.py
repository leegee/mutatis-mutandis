#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Event-log ingestion pipeline for EEBO embeddings.

This version implements a *true event model*:

Each emitted row corresponds to a single token occurrence event:
- A token in a document at a specific position
- Observed through overlapping transformer windows
- Aggregated from contextualised hidden states

Core invariant:
- Each stored row is an atomic semantic observation event
- Events are reconstructable without cross-column alignment assumptions
- Postgres remains source of truth for identity and lexical metadata

Important conceptual shift:
- We no longer store "parallel arrays of columns"
- We store *event objects* derived from model inference over windows
"""

from __future__ import annotations

import argparse
import shutil
import numpy as np
import torch

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EMBED_BATCH_SIZE
from lib.zarr_embedding_eventlog import ZarrEmbeddingEventLog
from lib.macberth import load_macberth, normalize


WINDOW_SIZE = 512
STRIDE = WINDOW_SIZE // 2


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

    if word_ids is None:
        word_ids = [None] * len(input_ids)

    return input_ids, attention_mask, word_ids


def iter_windows(input_ids, attention_mask, word_ids):
    n = len(input_ids)
    start = 0

    while start < n:
        end = min(start + WINDOW_SIZE, n)

        yield start, input_ids[start:end], attention_mask[start:end], word_ids[start:end]

        if end == n:
            break
        start += STRIDE


def forward(model, device, batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    input_ids = torch.tensor([pad(x["input_ids"]) for x in batch], dtype=torch.long).to(device)
    mask = torch.tensor([pad(x["mask"]) for x in batch], dtype=torch.long).to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=mask, return_dict=True)

    return out.last_hidden_state.cpu().numpy()


def process_doc(doc_id, tokens, vector_ids, tokenizer, model, device):
    input_ids, attention_mask, word_ids = encode_doc(tokens, tokenizer)

    dim = model.config.hidden_size

    # We will emit events directly, not accumulate per-token means
    events_vecs = []
    events_ids = []
    events_token_idx = []

    batch = []

    def emit_batch(batch):
        """
        Convert a batch of windows into per-token events.

        Each window produces contextualised embeddings.
        We do NOT aggregate across windows.
        """
        hidden = forward(model, device, batch)

        for b, h in zip(batch, hidden):
            wids = b["word_ids"]

            for i, wid in enumerate(wids):
                if wid is None or wid < 0:
                    continue

                # Each (window, token) interaction is a distinct event
                events_vecs.append(normalize(h[i]))
                events_ids.append(vector_ids[wid])
                events_token_idx.append(wid)

    for start, ids, mask, wids in iter_windows(input_ids, attention_mask, word_ids):
        batch.append({
            "input_ids": ids,
            "mask": mask,
            "word_ids": wids
        })

        if len(batch) >= EMBED_BATCH_SIZE:
            emit_batch(batch)
            batch.clear()

    if batch:
        emit_batch(batch)

    if not events_ids:
        return None

    vecs = np.asarray(events_vecs, dtype=np.float32)
    ids = np.asarray(events_ids, dtype=np.int64)
    token_idxs = np.asarray(events_token_idx, dtype=np.int64)

    return vecs, ids, token_idxs


def process_slice(conn, slice_range, tokenizer, model, device):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    store = ZarrEmbeddingEventLog(
        path=str(ZARR_ROOT / "tier1" / slice_id),
        dim=model.config.hidden_size,
    )

    cur = conn.cursor(name=f"tier1_{slice_id}")
    cur.itersize = 10000

    cur.execute("""
        SELECT doc_id, token_idx, vector_id, token
        FROM pamphlet_tokens
        WHERE pub_year BETWEEN %s AND %s
        ORDER BY doc_id, token_idx
    """, slice_range)

    current_doc = None
    buf_tokens = []
    buf_vids = []
    buf_doc_id = None

    def flush():
        nonlocal buf_tokens, buf_vids, buf_doc_id

        if not buf_tokens:
            return

        result = process_doc(
            buf_doc_id,
            buf_tokens,
            buf_vids,
            tokenizer,
            model,
            device,
        )

        if result is None:
            return

        vecs, ids, token_idxs = result

        store.append_events(
            emb_norm=vecs,
            emb_raw=vecs,
            vector_id=ids,
            doc_id=np.array([buf_doc_id] * len(ids), dtype="U32"),
            token_idx=token_idxs,
        )

    for doc_id, token_idx, vid, token in cur:

        if current_doc is None:
            current_doc = doc_id
            buf_doc_id = doc_id

        if doc_id != current_doc:
            flush()
            buf_tokens.clear()
            buf_vids.clear()
            current_doc = doc_id
            buf_doc_id = doc_id

        buf_tokens.append(token)
        buf_vids.append(vid)

    flush()

    logger.info(f"[SLICE COMPLETE] {slice_id}")


def clear_output_dir():
    path = ZARR_ROOT / "tier1"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-clear", action="store_true")
    p.add_argument("--first", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.no_clear:
        logger.info("Clearing Tier 1 output")
        clear_output_dir()

    conn = get_connection()
    mac = load_macberth()

    slices = SLICES[:1] if args.first else SLICES

    for s in slices:
        process_slice(conn, s, mac.tokenizer, mac.model, mac.device)

    conn.close()


if __name__ == "__main__":
    main()
