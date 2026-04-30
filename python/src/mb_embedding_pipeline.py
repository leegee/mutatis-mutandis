#!/usr/bin/env python
from __future__ import annotations

from typing import Optional, List, Tuple
import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from lib.eebo_vectors import save_vectors
from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import (
    SLICES,
    EEBO_MODEL_NAME,
    MACBERTH_FINE_TUNED_DIR
)
from lib.mb_paths import faiss_slice_path
from lib.FaissIndex import FaissIndex


SAVE_OCCURRENCE_VECTORS = os.getenv("SAVE_OCCURRENCE_VECTORS", "1") == "1"

TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None
_DEVICE: Optional[str] = None


# sliding window avoids truncation loss at 512
WINDOW_SIZE = 384
WINDOW_STRIDE = 256

# CPU-friendly batching
TOKEN_BUDGET = 12000


@dataclass
class SentenceBatchItem:
    doc_id: str
    sentence: str
    token_keys: List[Tuple[str, int, int]]  # (doc_id, token_idx, vector_id)


def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def normalize(v: np.ndarray) -> Optional[np.ndarray]:
    v = v.astype(np.float32, copy=False)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    return v / n


def get_macberth_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        logger.info("Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(EEBO_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(EEBO_MODEL_NAME)

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Tokenizer must be fast for offset alignment")

        ft_dir = MACBERTH_FINE_TUNED_DIR
        if all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"]):
            logger.info("Loading fine-tuned weights...")
            state_dict = torch.load(ft_dir / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        TOKENIZER, MODEL = tokenizer, model

    return TOKENIZER, MODEL


def forward_batch(model, tokenizer, sentences: List[str]):
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,     # safe because windowing guarantees fit
        max_length=512,
        return_offsets_mapping=True
    )

    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    inputs = {k: v.to(get_device()) for k, v in enc.items() if k != "offset_mapping"}

    outputs = model(**inputs, output_hidden_states=True)

    # invariant: representation space defined here only
    layers = outputs.hidden_states
    mixed = torch.stack(layers[2:-2], dim=0).mean(dim=0)

    hidden = mixed.detach().cpu().numpy()

    return hidden, offsets, input_ids, attention_mask


def process_token_batch(
    batch: List[SentenceBatchItem],
    model,
    tokenizer,
    index,
    vecs_accum,
    ids_accum,
):
    sentences = [b.sentence for b in batch]

    hidden, offsets, input_ids, attention_mask = forward_batch(
        model, tokenizer, sentences
    )

    special_ids = set(tokenizer.all_special_ids)

    for b_i, item in enumerate(batch):
        text = item.sentence
        keys = item.token_keys

        if not keys:
            continue

        spans = []
        cursor = 0
        tokens = text.split(" ")

        for tok in tokens:
            start = cursor
            end = start + len(tok)
            spans.append((start, end))
            cursor = end + 1

        token_vecs = hidden[b_i]
        token_offsets = offsets[b_i].tolist()
        ids = input_ids[b_i].tolist()
        mask = attention_mask[b_i].tolist()

        token_to_subwords = [[] for _ in spans]

        for sub_i, ((s, e), tok_id, m) in enumerate(zip(token_offsets, ids, mask)):
            if m == 0 or tok_id in special_ids or s == e:
                continue

            for t_i, (ts, te) in enumerate(spans):
                if not (e <= ts or s >= te):
                    token_to_subwords[t_i].append(sub_i)
                    break

        for i, (_, _, vector_id) in enumerate(keys):
            sub_idxs = token_to_subwords[i]
            if not sub_idxs:
                continue

            vec = normalize(np.mean([token_vecs[j] for j in sub_idxs], axis=0))
            if vec is None:
                continue

            vector_id = int(vector_id)
            index.add(vec.reshape(1, -1), [vector_id])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(vec)
                ids_accum.append(vector_id)



def stream_for_embedding(conn):
    """
    Invariant:
        tokens.vector_id is the stable identity.
        Ordering is irrelevant for identity, only for batching.
    """

    with conn.cursor(name="eebo_stream") as cur:
        cur.itersize = 10_000

        cur.execute("""
            SELECT
                doc_id,
                token_idx,
                token,
                vector_id
            FROM tokens
            ORDER BY doc_id, token_idx;
        """)

        current_doc = None
        buffer = []

        for doc_id, token_idx, token_text, vector_id in cur:
            if vector_id is None:
                raise ValueError(f"Missing vector_id for {doc_id}:{token_idx}")

            if doc_id != current_doc and buffer:
                yield from _emit_windows(current_doc, buffer)
                buffer.clear()

            current_doc = doc_id
            buffer.append((token_idx, token_text, vector_id))

        if buffer:
            yield from _emit_windows(current_doc, buffer)


def _emit_windows(doc_id, tokens):
    n = len(tokens)
    start = 0

    while start < n:
        end = min(start + WINDOW_SIZE, n)
        window = tokens[start:end]

        text_tokens = [t[1] for t in window]
        keys = [(doc_id, t[0], t[2]) for t in window]

        yield doc_id, " ".join(text_tokens), keys

        if end == n:
            break

        start += WINDOW_STRIDE


def batch_stream(stream):
    batch = []
    token_count = 0

    for doc_id, text, keys in stream:
        n_tokens = len(keys)

        if batch and token_count + n_tokens > TOKEN_BUDGET:
            yield batch
            batch = []
            token_count = 0

        batch.append(SentenceBatchItem(doc_id, text, keys))
        token_count += n_tokens

    if batch:
        yield batch


def process_slice(conn, slice_range):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"

    tokenizer, model = get_macberth_model()
    model.to(get_device())

    index = FaissIndex(model.config.hidden_size)

    vecs_accum = []
    ids_accum = []

    stream = stream_for_embedding(conn)

    with torch.no_grad():
        for batch in batch_stream(stream):
            process_token_batch(
                batch, model, tokenizer, index, vecs_accum, ids_accum
            )

    index.save(faiss_slice_path(slice_range))

    if SAVE_OCCURRENCE_VECTORS:
        save_vectors(slice_id, vecs_accum, ids_accum)

    logger.info(
        f"[SLICE COMPLETE] {slice_id} "
        f"vectors={len(ids_accum)} faiss={index._index.ntotal}"
    )


def build_all_slices():
    conn = get_connection()
    for s in SLICES:
        process_slice(conn, s)
    conn.close()


def main():
    logger.info("Starting embedding pipeline")
    build_all_slices()


if __name__ == "__main__":
    main()
