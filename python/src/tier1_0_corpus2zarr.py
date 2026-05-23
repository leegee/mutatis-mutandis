#!/usr/bin/env python
"""
tier1_corpus2zarr.py

Contextual Observation Log (Tier 1)

Core invariant
--------------
- Postgres defines corpus identity
- concept_id defines stable lexical occurrence identity
- event_id defines contextual embedding observation identity
- FAISS indexes event_id space ONLY
"""

from __future__ import annotations

import argparse
import shutil
import numpy as np
import torch
import xxhash

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import ZARR_ROOT, SLICES, EMBED_BATCH_SIZE
from lib.zarr_embedding_observation_store import ZarrEmbeddingObservationStore
from lib.macberth import load_macberth


WINDOW_SIZE = 512
STRIDE = WINDOW_SIZE // 2


def stable_hash(key: str) -> np.int64:
    h = xxhash.xxh64(key, seed=0).intdigest()
    return np.int64(h & 0x7FFFFFFFFFFFFFFF)


# Stable textual locus in corpus space.
# Must remain invariant across contextual embedding passes.
def make_concept_id(doc_id, token_idx):
    return stable_hash(f"{doc_id}:{token_idx}")


# Unique contextual embedding observation.
# One event_id MUST correspond to exactly one emitted vector.
def make_event_id(doc_id, token_idx, window_start, window_token_pos):
    return stable_hash(
        f"{doc_id}:{token_idx}:{window_start}:{window_token_pos}"
    )


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

        yield (
            start,
            input_ids[start:end],
            attention_mask[start:end],
            word_ids[start:end]
        )

        if end == n:
            break

        start += STRIDE


def forward(model, device, batch):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq):
        return seq + [0] * (max_len - len(seq))

    input_ids = torch.tensor(
        [pad(x["input_ids"]) for x in batch],
        dtype=torch.long
    ).to(device)

    mask = torch.tensor(
        [pad(x["mask"]) for x in batch],
        dtype=torch.long
    ).to(device)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            return_dict=True
        )

    return out.last_hidden_state.cpu().numpy()


def process_windows_events(
    doc_id,
    tokens,
    vector_ids,
    tokenizer,
    model,
    device
):
    input_ids, attention_mask, word_ids = encode_doc(tokens, tokenizer)

    events = []
    batch = []

    def flush(batch):
        if not batch:
            return []

        hidden = forward(model, device, batch)

        out = []

        for b, h in zip(batch, hidden):
            out.extend(
                extract_events(
                    doc_id,
                    tokens,
                    vector_ids,
                    b,
                    h
                )
            )

        return out

    for window_start, ids, mask, wids in iter_windows(
        input_ids,
        attention_mask,
        word_ids
    ):
        batch.append({
            "input_ids": ids,
            "mask": mask,
            "word_ids": wids,
            "window_start": window_start
        })

        if len(batch) >= EMBED_BATCH_SIZE:
            events.extend(flush(batch))
            batch.clear()

    if batch:
        events.extend(flush(batch))

    return events if events else None


def extract_events(doc_id, tokens, vector_ids, batch_item, hidden):
    window_start = batch_item["window_start"]
    word_ids = batch_item["word_ids"]

    events = []

    for i, wid in enumerate(word_ids):
        if wid is None or wid < 0:
            continue

        concept_id = make_concept_id(
            doc_id,
            int(wid)
        )

        event_id = make_event_id(
            doc_id,
            int(wid),
            int(window_start),
            int(i)
        )

        events.append((
            event_id,                  # unique contextual observation
            concept_id,               # stable corpus token identity
            doc_id,
            int(wid),                 # corpus token position
            int(window_start),        # contextual frame origin
            int(i),                   # intra-window transformer position
            tokens[wid],
            int(vector_ids[wid]),
            hidden[i].astype(np.float32)
        ))

    return events


def process_slice(conn, slice_range, tokenizer, model, device):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"

    logger.info(f"[SLICE START] {slice_id}")

    store = ZarrEmbeddingObservationStore(
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

        result = process_windows_events(
            buf_doc_id,
            buf_tokens,
            buf_vids,
            tokenizer,
            model,
            device,
        )

        if result is None:
            return

        event_ids = []
        concept_ids = []
        doc_ids = []
        token_idxs = []
        window_ids = []
        window_pos = []
        tokens_out = []
        vector_ids_out = []
        vecs = []

        for (
            eid,
            cid,
            d_id,
            t_idx,
            w_id,
            w_pos,
            tok,
            v_id,
            vec
        ) in result:
            event_ids.append(eid)
            concept_ids.append(cid)
            doc_ids.append(d_id)
            token_idxs.append(t_idx)
            window_ids.append(w_id)
            window_pos.append(w_pos)
            tokens_out.append(tok)
            vector_ids_out.append(v_id)
            vecs.append(vec)

        store.append_events(
            event_id=np.asarray(event_ids, dtype=np.int64),
            concept_id=np.asarray(concept_ids, dtype=np.int64),
            emb_raw=np.stack(vecs),
            vector_id=np.asarray(vector_ids_out, dtype=np.int64),
            doc_id=np.asarray(doc_ids, dtype="U32"),
            token_idx=np.asarray(token_idxs, dtype=np.int32),
            window_id=np.asarray(window_ids, dtype=np.int32),
            window_token_pos=np.asarray(window_pos, dtype=np.int32),
            token=np.asarray(tokens_out, dtype=object),
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
        process_slice(
            conn,
            s,
            mac.tokenizer,
            mac.model,
            mac.device
        )

    conn.close()


if __name__ == "__main__":
    main()
