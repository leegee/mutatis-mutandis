#!/usr/bin/env python

from __future__ import annotations
from typing import Optional, List, Tuple
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



def forward_single(model, tokenizer, text: str):
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        return_offsets_mapping=True
    )

    input_ids = enc["input_ids"]
    max_len = model.config.max_position_embeddings

    seq_len = input_ids.shape[1]

    if seq_len > max_len:
        raise ValueError(f"Sequence too long: {seq_len} > {max_len}")

    inputs = {k: v.to(get_device()) for k, v in enc.items() if k != "offset_mapping"}
    outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1][0].detach().cpu().numpy()
    offsets = enc["offset_mapping"][0].tolist()

    return hidden, offsets


def process_single_sentence(
    item: SentenceBatchItem,
    model,
    tokenizer
):
    text = item.sentence
    tokens = text.split(" ")
    occ_ids = item.token_occurrence_ids
    max_len = model.config.max_position_embeddings

    try:
        hidden, offsets = forward_single(model, tokenizer, text)
        return [(item, hidden, offsets)]
    except ValueError as e:
        first_id = occ_ids[0] if occ_ids else None
        last_id  = occ_ids[-1] if occ_ids else None
        logger.warning(
            f"[doc_id={item.doc_id} occ={first_id}-{last_id}] "
            f"{str(e)} | chars={len(text)}\n"
            f"{text[:300]}"
        )

        chunks = []
        chunk_tokens = []
        chunk_ids = []

        for tok, oid in zip(tokens, occ_ids, strict=Trie):
            candidate_tokens = chunk_tokens + [tok]
            candidate_text = " ".join(candidate_tokens)

            test_ids = tokenizer(
                candidate_text,
                add_special_tokens=True
            )["input_ids"]

            if len(test_ids) > max_len:
                if chunk_tokens:
                    chunks.append((chunk_tokens, chunk_ids))
                    chunk_tokens = [tok]
                    chunk_ids = [oid]
                else:
                    # single token too large → force split at subword level
                    sub_ids = tokenizer(tok, add_special_tokens=True)["input_ids"]

                    if len(sub_ids) > max_len:
                        raise ValueError(
                            f"Single token explodes beyond model limit: '{tok}'"
                        )

                    chunks.append(([tok], [oid]))
                    chunk_tokens = []
                    chunk_ids = []
            else:
                chunk_tokens.append(tok)
                chunk_ids.append(oid)

        if chunk_tokens:
            chunks.append((chunk_tokens, chunk_ids))

        outputs = []

        for toks, ids in chunks:
            sub_text = " ".join(toks)

            # HARD GUARD before model call
            test_ids = tokenizer(sub_text, add_special_tokens=True)["input_ids"]
            if len(test_ids) > max_len:
                raise ValueError(
                    f"Chunk still too large after splitting: {len(test_ids)} > {max_len}"
                )

            hidden, offsets = forward_single(model, tokenizer, sub_text)

            outputs.append((
                SentenceBatchItem(item.doc_id, sub_text, ids),
                hidden,
                offsets
            ))

        return outputs


def process_token_batch(
    batch: List[SentenceBatchItem],
    model,
    tokenizer,
    index: FaissIndex,
    vecs_accum: List[np.ndarray],
    ids_accum: List[int],
):
    results = []

    for item in batch:
        results.extend(process_single_sentence(item, model, tokenizer))

    for item, token_vecs, offsets_i in results:

        text = item.sentence
        occ_ids = item.token_occurrence_ids

        tokens = text.split(" ")
        spans = []

        cursor = 0
        for tok in tokens:
            start = cursor
            end = start + len(tok)
            spans.append((start, end))
            cursor = end + 1

        if len(spans) != len(occ_ids):
            raise ValueError("Token/span mismatch")

        token_to_vecs = [[] for _ in spans]

        for sub_idx, (start, end) in enumerate(offsets_i):
            if start == end:
                continue

            for tok_idx, (t_start, t_end) in enumerate(spans):
                if start >= t_start and end <= t_end:
                    token_to_vecs[tok_idx].append(token_vecs[sub_idx])
                    break

        for tok_idx, occ_id in enumerate(occ_ids):
            sub_vecs = token_to_vecs[tok_idx]

            if not sub_vecs:
                raise ValueError(f"No vectors for token {occ_id}")

            vec = np.mean(sub_vecs, axis=0)
            v = normalize(vec)
            if v is None:
                continue

            index.add(v.reshape(1, -1), [occ_id])

            if SAVE_OCCURRENCE_VECTORS:
                vecs_accum.append(v)
                ids_accum.append(occ_id)

    batch.clear()
    gc.collect()


def process_slice(conn, slice_range, batch_size=128):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"

    tokenizer, model = get_macberth_model()
    model.to(get_device())

    index = FaissIndex(model.config.hidden_size)

    vecs_accum: List[np.ndarray] = []
    ids_accum: List[int] = []

    stream = stream_sentences_within_model_limit(conn, slice_range, tokenizer, model.config.max_position_embeddings)

    batch: List[SentenceBatchItem] = []

    with torch.no_grad():
        for doc_id, sent, occ_ids in stream:
            batch.append(SentenceBatchItem(doc_id, sent, occ_ids))

            if len(batch) >= batch_size:
                process_token_batch(batch, model, tokenizer, index, vecs_accum, ids_accum)

        if batch:
            process_token_batch(batch, model, tokenizer, index, vecs_accum, ids_accum)

    if SAVE_OCCURRENCE_VECTORS and len(ids_accum) != index._index.ntotal:
        raise ValueError("FAISS size mismatch")

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
