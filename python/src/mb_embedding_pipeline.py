#!/usr/bin/env python
from __future__ import annotations

from typing import Optional, List, Tuple
import os
import re
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

WINDOW_SIZE = 384
WINDOW_STRIDE = WINDOW_SIZE
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


def get_valid_subwords(offsets):
    # removes special tokens like [CLS], [SEP]
    return [
        (i, s, e)
        for i, (s, e) in enumerate(offsets)
        if not (s == 0 and e == 0)
    ]


def build_token_spans(text: str):
    return [
        (m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]


def overlaps(a_s, a_e, b_s, b_e):
    return not (a_e <= b_s or a_s >= b_e)


def map_tokens_to_subwords(text, offsets):
    spans = build_token_spans(text)
    valid_subwords = get_valid_subwords(offsets)

    token_to_subwords = [[] for _ in spans]

    for sub_i, s, e in valid_subwords:
        for t_i, (ts, te) in enumerate(spans):
            if overlaps(s, e, ts, te):
                token_to_subwords[t_i].append(sub_i)
                break

    return token_to_subwords, spans


def forward_batch(device, model, tokenizer, sentences: List[str]):
    enc = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    offsets = enc["offset_mapping"]
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    inputs = {k: v.to(device) for k, v in enc.items() if k != "offset_mapping"}

    outputs = model(**inputs, output_hidden_states=True)

    layers = outputs.hidden_states
    mixed = torch.stack(layers[2:-2], dim=0).mean(dim=0)

    hidden = mixed.detach().cpu().numpy()

    return hidden, offsets, input_ids, attention_mask


def process_token_batch(
    device,
    batch: List[SentenceBatchItem],
    model,
    tokenizer,
    index,
    vecs_accum,
    ids_accum,
):
    sentences = [b.sentence for b in batch]

    hidden, offsets, input_ids, attention_mask = forward_batch(
        device, model, tokenizer, sentences
    )

    special_ids = set(tokenizer.all_special_ids)

    for b_i, item in enumerate(batch):
        text = item.sentence
        keys = item.token_keys  # (doc_id, token_idx, vector_id)

        if not keys:
            continue

        # OFFSET-DRIVEN ALIGNMENT (single source of truth)
        token_to_subwords, spans = map_tokens_to_subwords(
            text,
            offsets[b_i].tolist()
        )

        token_vecs = hidden[b_i]
        ids = input_ids[b_i].tolist()
        mask = attention_mask[b_i].tolist()

        # VECTOR CONSTRUCTION + FAISS INSERT
        for i, (_, _, vector_id) in enumerate(keys):
            sub_idxs = token_to_subwords[i]

            if not sub_idxs:
                continue

            vec = normalize(
                np.mean([token_vecs[j] for j in sub_idxs], axis=0)
            )

            if vec is None:
                continue

            vector_id = int(vector_id)

            index.add(vec.reshape(1, -1), [vector_id])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(vec)
                ids_accum.append(vector_id)


def stream_for_embedding(conn):
    """
    vector_id is the ONLY identity layer.

    Invariant:
        tokens.vector_id must be pre-populated before embedding.
    """

    with conn.cursor(name="eebo_stream") as cur:
        cur.itersize = 10_000

        cur.execute("""
            SELECT
                doc_id,
                token_idx,
                token,
                vector_id
            FROM pamphlet_tokens
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


def embed_document(
    device,
    doc_id: str,
    text: str,
    vector_keys,
    model,
    tokenizer,
    index,
    vecs_accum,
    ids_accum,
):
    """
    Option B core:
    full document → model → hidden states → window in embedding space
    """

    hidden, offsets, input_ids, attention_mask = forward_batch(
        device, model, tokenizer, [text]
    )

    hidden = hidden[0]
    offsets = offsets[0].tolist()

    special_ids = set(tokenizer.all_special_ids)

    token_to_subwords, _ = map_tokens_to_subwords(text, offsets)

    # FULL-DOCUMENT EMBEDDING SPACE WINDOWING
    n_tokens = len(vector_keys)
    start = 0

    while start < n_tokens:
        end = min(start + WINDOW_SIZE, n_tokens)

        window_keys = vector_keys[start:end]

        window_vecs = []

        for i, (_, _, vector_id) in enumerate(window_keys):
            sub_idxs = token_to_subwords[start + i] if (start + i) < len(token_to_subwords) else []

            if not sub_idxs:
                continue

            vec = normalize(
                np.mean([hidden[j] for j in sub_idxs], axis=0)
            )

            if vec is None:
                continue

            vector_id = int(vector_id)

            index.add(vec.reshape(1, -1), [vector_id])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(vec)
                ids_accum.append(vector_id)

        if end == n_tokens:
            break

        start += WINDOW_STRIDE


def _emit_windows(doc_id, tokens):
    """
    tokens: List[(token_idx, token_text, vector_id)]

    Character-based windowing with strict token assignment.

    Invariants:
    - Each token is emitted exactly once
    - Window text matches tokenizer offsets
    - Token→window assignment is based on span midpoint

    To do:
    - Semantic awareness (sentence and clauses)
    """

    # reconstruct full text
    text_tokens = [t[1] for t in tokens]
    vector_keys = [(doc_id, t[0], t[2]) for t in tokens]

    text = " ".join(text_tokens)

    # build spans once (global coordinate system)
    spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    assert len(spans) == len(vector_keys), "span/token mismatch"

    n_chars = len(text)

    start = 0

    while start < n_chars:
        end = min(start + WINDOW_SIZE, n_chars)

        window_text = text[start:end]

        window_keys = []

        for i, (s, e) in enumerate(spans):
            # assign token to window by midpoint (prevents duplication)
            mid = (s + e) // 2

            if start <= mid < end:
                window_keys.append(vector_keys[i])

        if window_keys:
            yield doc_id, window_text, window_keys

        if end == n_chars:
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
    device = get_device()
    model.to(device)

    index = FaissIndex(model.config.hidden_size)

    vecs_accum = []
    ids_accum = []

    with conn.cursor(name="eebo_stream") as cur:
        cur.itersize = 10_000

        cur.execute("""
            SELECT
                doc_id,
                token_idx,
                token,
                vector_id
            FROM pamphlet_tokens
            ORDER BY doc_id, token_idx;
        """)

        current_doc = None
        buffer = []

        for doc_id, token_idx, token_text, vector_id in cur:
            if vector_id is None:
                raise ValueError(f"Missing vector_id for {doc_id}:{token_idx}")

            if doc_id != current_doc and buffer:
                text_tokens = [t[1] for t in buffer]
                vector_keys = [(doc_id, t[0], t[2]) for t in buffer]

                text = " ".join(text_tokens)

                embed_document(
                    device,
                    doc_id,
                    text,
                    vector_keys,
                    model,
                    tokenizer,
                    index,
                    vecs_accum,
                    ids_accum,
                )

                buffer.clear()

            current_doc = doc_id
            buffer.append((token_idx, token_text, vector_id))

        if buffer:
            text_tokens = [t[1] for t in buffer]
            vector_keys = [(current_doc, t[0], t[2]) for t in buffer]

            text = " ".join(text_tokens)

            embed_document(
                device,
                current_doc,
                text,
                vector_keys,
                model,
                tokenizer,
                index,
                vecs_accum,
                ids_accum,
            )

    if SAVE_OCCURRENCE_VECTORS and len(set(ids_accum)) != len(ids_accum):
        raise ValueError("Duplicate vector_id in accumulation buffer")

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
