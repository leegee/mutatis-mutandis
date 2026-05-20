#!/usr/bin/env python
"""
mb_embedding_pipeline.py - Window-consistent embedding and indexing pipeline for EEBO pamphlet tokens.

This script streams tokenized EEBO documents from the database, reconstructs each
document, and generates contextual token embeddings using MacBERTh.
It aggregates subword representations back to token-level vectors, normalizes them,
and incrementally builds a FAISS index for efficient similarity search.

- Streaming ingestion:
  Documents are processed sequentially via a server-side cursor to avoid loading
  the full corpus into memory.

- Windowed encoding:
  Long documents are split into overlapping windows (WINDOW_SIZE, WINDOW_STRIDE)
  to preserve contextual continuity while respecting model input limits.

- Subword → token projection:
  Fast tokenizer offset mappings are used to align subword embeddings back to
  original tokens, ensuring token-level representations remain consistent across
  window boundaries.

- Layer aggregation:
  Hidden states from intermediate transformer layers are averaged to produce
  stable contextual embeddings.

- Vector accumulation and normalization:
  Multiple occurrences of a token within a document are averaged and L2-normalized
  before insertion into the index.

- Incremental FAISS indexing:
  Vectors are added in batches to a per-slice FAISS index, enabling scalable
  construction over large corpora.

- Checkpointing and fault tolerance:
  Progress is periodically checkpointed (by document ID and index state) to allow
  safe resumption after interruption. Index writes are atomic to prevent corruption.

- Optional occurrence persistence:
  Final token vectors can also be persisted separately for downstream analysis
  (eg semantic drift studies).

Data assumptions and invariants:

- Input rows must be ordered by (doc_id, token_idx).
- Each token must have a stable vector_id.
- Tokenization used here must align with the stored token sequence (whitespace-joined).
- Fast tokenizer with offset mappings is required.

Failure modes:

- Misalignment between tokenizer offsets and original tokens will silently degrade
  embedding quality.
- Missing vector_id raises a hard error.
- Partial checkpoint writes are avoided via atomic file replacement, but external
  interruptions between index and checkpoint saves may cause minor duplication
  on resume.

ENVIRONMENT:

- SAVE_OCCURRENCE_VECTORS: toggle persistence of per-token vectors
- CHECKPOINT_EVERY_DOCS: checkpoint frequency
- CHECKPOINT_DIR: directory for checkpoint files
- MAX_TEST_DOCS: limit for test runs

Outputs:

- FAISS index per slice
- Optional vector dumps per slice
- Checkpoint files for resumption
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict
import os
import json
from collections import defaultdict

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
    MACBERTH_MODEL_PATH,
    MACBERTH_FINE_TUNED_DIR
)
from lib.mb_paths import faiss_slice_path
from lib.FaissIndex import FaissIndex


SAVE_OCCURRENCE_VECTORS = os.getenv("SAVE_OCCURRENCE_VECTORS", "1") == "1"
CHECKPOINT_EVERY_DOCS = int(os.getenv("CHECKPOINT_EVERY_DOCS", "200"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
MAX_TEST_DOCS = int(os.getenv("MAX_TEST_DOCS", "0"))

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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


def get_macberth_model():
    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        logger.info("Loading model...")

        tokenizer = AutoTokenizer.from_pretrained(MACBERTH_MODEL_PATH, local_files_only=True)
        model = AutoModelForMaskedLM.from_pretrained(MACBERTH_MODEL_PATH, local_files_only=True)

        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError("Tokenizer must be fast")

        ft_dir = MACBERTH_FINE_TUNED_DIR
        if all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"]):
            logger.info("Loading fine-tuned weights...")
            state_dict = torch.load(ft_dir / "pytorch_model.bin", map_location="cpu")
            model.load_state_dict(state_dict, strict=False)

        model.eval()

        TOKENIZER, MODEL = tokenizer, model

    return TOKENIZER, MODEL


def atomic_save(path, save_fn):
    tmp = f"{path}.tmp"
    save_fn(tmp)
    os.replace(tmp, path)


def checkpoint_path(slice_id):
    return os.path.join(CHECKPOINT_DIR, f"{slice_id}.json")


def save_checkpoint(slice_id, last_doc_id):
    with open(checkpoint_path(slice_id), "w") as f:
        json.dump({"last_doc_id": last_doc_id}, f)


def load_checkpoint(slice_id):
    p = checkpoint_path(slice_id)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)["last_doc_id"]


def save_index_atomic(index, path):
    atomic_save(path, lambda p: index.save(p))


def project_subwords_to_tokens(
    offsets: List[Tuple[int, int]],
    token_count: int,
) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = {i: [] for i in range(token_count)}

    if not offsets:
        return mapping

    sub_i = 0
    S = len(offsets)

    for tok_i in range(token_count):
        start = sub_i

        while sub_i < S:
            s, e = offsets[sub_i]

            if s == 0 and e == 0:
                sub_i += 1
                continue

            if sub_i > start and s > offsets[start][1]:
                break

            mapping[tok_i].append(sub_i)
            sub_i += 1

    return mapping


def embed_document(
    device,
    doc_id: str,
    tokens: List[str],
    vector_keys,
    model,
    tokenizer,
    vec_acc: Dict[int, List[np.ndarray]],
):
    text = " ".join(tokens)

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

    token_count = len(tokens)

    for w_i in range(len(hidden)):
        window_hidden = hidden[w_i]
        window_offsets = offsets[w_i].tolist()

        subword_map = project_subwords_to_tokens(
            window_offsets,
            token_count
        )

        for tok_i, (_, _, vector_id) in enumerate(vector_keys):
            sub_idxs = subword_map.get(tok_i)
            if not sub_idxs:
                continue

            vec = np.mean([window_hidden[j] for j in sub_idxs], axis=0)
            vec_acc[int(vector_id)].append(vec)


def flush_vectors(
    index: FaissIndex,
    vec_acc: Dict[int, List[np.ndarray]],
    vecs_accum: List[np.ndarray],
    ids_accum: List[int],
):
    if not vec_acc:
        return

    batch_vecs = []
    batch_ids = []

    for vid, vecs in vec_acc.items():
        stacked = np.stack(vecs, axis=0)
        final_vec = normalize(np.mean(stacked, axis=0))

        if final_vec is None:
            continue

        batch_vecs.append(final_vec)
        batch_ids.append(vid)

        if SAVE_OCCURRENCE_VECTORS:
            vecs_accum.append(final_vec)
            ids_accum.append(vid)

    if batch_vecs:
        vec_matrix = np.vstack(batch_vecs).astype(np.float32)
        index.add(vec_matrix, batch_ids)

    vec_acc.clear()


def process_slice(conn, slice_range):
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"[SLICE START] {slice_id}")

    tokenizer, model = get_macberth_model()
    device = get_device()
    model.to(device)

    index = FaissIndex(model.config.hidden_size)

    vec_acc: Dict[int, List[np.ndarray]] = defaultdict(list)
    vecs_accum = []
    ids_accum = []

    last_done = load_checkpoint(slice_id)
    skipping = last_done is not None

    docs_processed = 0

    def flush(doc_id, buffer):
        if not buffer:
            return

        tokens = [t[1] for t in buffer]
        vector_keys = [(doc_id, t[0], t[2]) for t in buffer]

        embed_document(
            device,
            doc_id,
            tokens,
            vector_keys,
            model,
            tokenizer,
            vec_acc,
        )

        flush_vectors(index, vec_acc, vecs_accum, ids_accum)

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

            if skipping:
                if doc_id == last_done:
                    skipping = False
                continue

            if vector_id is None:
                raise ValueError(f"Missing vector_id {doc_id}:{token_idx}")

            if current_doc is not None and doc_id != current_doc:
                flush(current_doc, buffer)
                buffer.clear()

                docs_processed += 1

                if docs_processed % CHECKPOINT_EVERY_DOCS == 0:
                    logger.info(f"[CHECKPOINT] {slice_id} doc={current_doc}")

                    save_index_atomic(index, faiss_slice_path(slice_range))
                    save_checkpoint(slice_id, current_doc)

                    if SAVE_OCCURRENCE_VECTORS:
                        save_vectors(slice_id, vecs_accum, ids_accum)

                    vecs_accum.clear()
                    ids_accum.clear()

            current_doc = doc_id
            buffer.append((token_idx, token_text, vector_id))
            if MAX_TEST_DOCS and docs_processed >= MAX_TEST_DOCS:
                logger.info(f"[TEST STOP] reached MAX_TEST_DOCS={MAX_TEST_DOCS}")
                break

        if buffer:
            flush(current_doc, buffer)

    flush_vectors(index, vec_acc, vecs_accum, ids_accum)

    save_index_atomic(index, faiss_slice_path(slice_range))
    save_checkpoint(slice_id, current_doc)

    if SAVE_OCCURRENCE_VECTORS:
        save_vectors(slice_id, vecs_accum, ids_accum)

    logger.info(
        f"[SLICE COMPLETE] {slice_id} "
        f"vectors={len(ids_accum)} ntotal={index._index.ntotal}"
    )


def build_all_slices():
    conn = get_connection()
    for s in SLICES:
        process_slice(conn, s)
    conn.close()


def main():
    logger.info("Starting window-consistent FAISS batching pipeline")
    build_all_slices()


if __name__ == "__main__":
    main()
