#!/usr/bin/env python
from __future__ import annotations

from typing import Optional, List, Tuple, Dict
import os

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

WINDOW_SIZE = 512
WINDOW_STRIDE = 256


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



# STRICT ALIGNMENT


def build_db_token_spans(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
    """
    Deterministic span reconstruction.

    Invariant:
    - tokens must appear sequentially in text with monotonic offsets
    - any deviation is treated as a hard failure
    """
    spans: List[Tuple[int, int]] = []
    cursor = 0

    for tok in tokens:
        idx = text.find(tok, cursor)

        if idx == -1:
            raise ValueError(f"Token not found in text at cursor={cursor}: {tok!r}")

        if idx < cursor:
            raise ValueError(f"Non-monotonic alignment for token={tok!r}")

        spans.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    return spans


def map_subwords_to_db_tokens(
    offsets: List[Tuple[int, int]],
    db_spans: List[Tuple[int, int]]
) -> Dict[int, List[int]]:
    """
    Linear-time mapping: O(S + T)

    Assumes:
    - offsets sorted by subword index
    - db_spans sorted by token index
    - both are monotonic in character space (true if reconstruction is consistent)
    """

    mapping: Dict[int, List[int]] = {i: [] for i in range(len(db_spans))}

    t = 0  # pointer into db_spans
    T = len(db_spans)

    for sub_i, (s, e) in enumerate(offsets):

        if s == 0 and e == 0:
            continue

        # advance db pointer until span could overlap
        while t < T and db_spans[t][1] <= s:
            t += 1

        # check current and next few tokens only (usually 1–2)
        probe = t

        while probe < T and db_spans[probe][0] < e:
            ts, te = db_spans[probe]

            overlap = max(0, min(e, te) - max(s, ts))

            if overlap > 0:
                mapping[probe].append(sub_i)

            probe += 1

            # early exit if we've passed the subword range
            if ts > e:
                break

    return mapping


# EMBEDDING CORE


def embed_document(
    device,
    doc_id: str,
    text: str,
    tokens: List[str],
    vector_keys,
    model,
    tokenizer,
    index,
    vecs_accum,
    ids_accum,
):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    offsets = enc["offset_mapping"]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    hidden = torch.stack(outputs.hidden_states[2:-2], dim=0).mean(dim=0)
    hidden = hidden.detach().cpu().numpy()

    db_spans = build_db_token_spans(text, tokens)

    for w_i in range(len(hidden)):
        window_hidden = hidden[w_i]
        window_offsets = offsets[w_i].tolist()

        subword_map = map_subwords_to_db_tokens(window_offsets, db_spans)

        for tok_i, (_, _, vector_id) in enumerate(vector_keys):

            sub_idxs = subword_map.get(tok_i)
            if not sub_idxs:
                continue

            vec = normalize(
                np.mean([window_hidden[j] for j in sub_idxs], axis=0)
            )

            if vec is None:
                continue

            vector_id = int(vector_id)

            index.add(vec.reshape(1, -1), [vector_id])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(vec)
                ids_accum.append(vector_id)



# PIPELINE


def process_slice(conn, slice_range):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    tokenizer, model = get_macberth_model()
    device = get_device()
    model.to(device)

    index = FaissIndex(model.config.hidden_size)

    vecs_accum = []
    ids_accum = []

    def flush_doc(doc_id, buffer):
        if not buffer:
            return

        tokens = [t[1] for t in buffer]
        vector_keys = [(doc_id, t[0], t[2]) for t in buffer]
        text = " ".join(tokens)

        embed_document(
            device,
            doc_id,
            text,
            tokens,
            vector_keys,
            model,
            tokenizer,
            index,
            vecs_accum,
            ids_accum,
        )

    with conn.cursor(name="eebo_stream") as cur:
        cur.itersize = 10_000

        cur.execute("""
            SELECT doc_id, token_idx, token, vector_id
            FROM pamphlet_tokens
            ORDER BY doc_id, token_idx;
        """)

        current_doc = None
        buffer = []

        for doc_id, token_idx, token_text, vector_id in cur:
            if vector_id is None:
                raise ValueError(f"Missing vector_id for {doc_id}:{token_idx}")

            if current_doc is not None and doc_id != current_doc:
                flush_doc(current_doc, buffer)
                buffer.clear()

            current_doc = doc_id
            buffer.append((token_idx, token_text, vector_id))

        if buffer:
            flush_doc(current_doc, buffer)

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
