#!/usr/bin/env python
"""
tier1_corpus_embed.py

Tier 1:
    - embed ALL tokens
    - no concept mapping
    - no aggregation
    - append-only numeric store
"""

from __future__ import annotations

from typing import List, Optional
import argparse
import shutil

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EEBO_MODEL_NAME
from lib.vector_store_zarr import ZarrVectorStore
from lib.embed import embed_window
from lib.macberth import load_macberth, normalize

WINDOW_SIZE = 512

_DEVICE = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clear-output", action="store_true")
    p.add_argument("--first-slice-only", action="store_true")
    return p.parse_args()


def clear_output_dir():
    path = ZARR_ROOT / "tier1"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def process_slice(conn, slice_range):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    mac = load_macberth()
    tokenizer = mac.tokenizer
    model = mac.model
    device = mac.device

    store = ZarrVectorStore(
        path=str(ZARR_ROOT / "tier1" / slice_id),
        dim=model.config.hidden_size
    )

    cur = conn.cursor(name=f"tier1_{slice_id}")
    cur.itersize = 10000

    cur.execute("""
        SELECT t.doc_id, t.token_idx, t.vector_id, t.token
        FROM tokens t
        JOIN documents d ON d.doc_id = t.doc_id
        WHERE d.pub_year BETWEEN %s AND %s
        ORDER BY t.doc_id, t.token_idx
    """, slice_range)

    current_doc = None
    buffer_tokens: List[str] = []
    buffer_ids: List[Optional[int]] = []

    def flush_doc():
        if not buffer_tokens:
            return

        enc = tokenizer(
            buffer_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=WINDOW_SIZE,
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

        vecs = []
        ids = []

        seq_len = min(len(buffer_tokens), hidden.shape[0])

        for i in range(seq_len):
            vid = safe_int(buffer_ids[i])
            if vid is None:
                continue

            vec = normalize(hidden[i])
            if vec is None:
                continue

            vecs.append(vec)
            ids.append(vid)

        if vecs:
            store.append(np.asarray(vecs), np.asarray(ids))

    for doc_id, token_idx, vector_id, token in cur:

        if current_doc is None:
            current_doc = doc_id

        if doc_id != current_doc:
            flush_doc()
            buffer_tokens.clear()
            buffer_ids.clear()
            current_doc = doc_id

        buffer_tokens.append(token)
        buffer_ids.append(vector_id)

    if buffer_tokens:
        flush_doc()

    cur.close()

    logger.info(f"[SLICE COMPLETE] {slice_id}")


def main():
    args = parse_args()

    if args.clear_output:
        logger.info("Clearing Tier 1 output")
        clear_output_dir()

    conn = get_connection()

    if args.first_slice_only:
        process_slice(conn, SLICES[0])
    else:
        for s in SLICES:
            process_slice(conn, s)

    conn.close()


if __name__ == "__main__":
    main()
