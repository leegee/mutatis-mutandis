#!/usr/bin/env python

from __future__ import annotations
from typing import Optional, List, Tuple, cast
import os
import gc
from dataclasses import dataclass

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import (
    SLICES,
    EEBO_MODEL_NAME,
    MACBERTH_FINE_TUNED_DIR
)
from lib.mb_paths import vectors_path, faiss_slice_path
from lib.eebo_sentences import stream_sentences_within_model_limit
from lib.FaissIndex import FaissIndex


SAVE_OCCURRENCE_VECTORS = os.getenv("SAVE_OCCURRENCE_VECTORS", "1") == "1"


TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None
_DEVICE: Optional[str] = None


@dataclass
class SentenceBatchItem:
    doc_id: str
    sentence: str
    token_occurrence_ids: List[int]


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


def save_vectors(slice_id: str, vecs: List[np.ndarray], ids: List[int]) -> None:
    path = vectors_path(slice_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(vecs) != len(ids):
        raise ValueError("vecs and ids must align")

    np.savez_compressed(
        path,
        vecs=np.stack(vecs).astype(np.float32),
        ids=np.array(ids, dtype=np.int64),
    )

    logger.info(f"Saved token-level vectors at {path}")


def get_macberth_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        logger.info("Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(EEBO_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(EEBO_MODEL_NAME)

        ft_dir = MACBERTH_FINE_TUNED_DIR
        if all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"]):
            logger.info("Loading fine-tuned weights...")
            state_dict = torch.load(ft_dir / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        TOKENIZER, MODEL = tokenizer, model

    return TOKENIZER, MODEL


def forward_batch(model, tokenizer, batch):
    enc = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    inputs = {k: v.to(get_device()) for k, v in enc.items() if k != "offset_mapping"}
    outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1].detach().cpu().numpy()
    return hidden, enc


def process_token_batch(
    batch: List[SentenceBatchItem],
    model,
    tokenizer,
    index: FaissIndex,
    vecs_accum: List[np.ndarray],
    ids_accum: List[int],
):
    sentences = [b.sentence for b in batch]

    hidden, enc = forward_batch(model, tokenizer, sentences)

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    for b_i, item in enumerate(batch):

        toks = input_ids[b_i].tolist()
        occ_ids = item.token_occurrence_ids

        token_vecs = hidden[b_i]

        # HARD ALIGNMENT INVARIANT:
        # one vector per occurrence ID
        n = min(len(toks), len(occ_ids), len(token_vecs))

        for i in range(n):
            vec = token_vecs[i]
            vid = occ_ids[i]

            v = normalize(vec)
            if v is None:
                continue

            index.add(v.reshape(1, -1), [vid])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(v)
                ids_accum.append(vid)

    batch.clear()
    gc.collect()


def process_slice(conn, slice_range, batch_size=128):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"

    tokenizer, model = get_macberth_model()
    model.to(get_device())

    index = FaissIndex(model.config.hidden_size)

    vecs_accum: List[np.ndarray] = []
    ids_accum: List[int] = []

    stream = stream_sentences_within_model_limit(conn, slice_range, tokenizer)

    batch: List[SentenceBatchItem] = []

    with torch.no_grad():
        for doc_id, sent, occ_ids in stream:

            batch.append(SentenceBatchItem(doc_id, sent, occ_ids))

            if len(batch) >= batch_size:
                process_token_batch(batch, model, tokenizer, index, vecs_accum, ids_accum)

        if batch:
            process_token_batch(batch, model, tokenizer, index, vecs_accum, ids_accum)

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
    logger.info("Starting token-level pipeline")
    build_all_slices()


if __name__ == "__main__":
    main()
