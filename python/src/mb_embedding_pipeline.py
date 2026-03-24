#!/usr/bin/env python
"""
mb_embedding_pipeline.py

Generate token embeddings per slice (MacBERTh per-slice models) and build FAISS indexes.
- MacBERTh: embeddings per slice
- FAISS: always loads saved embeddings, flattens, normalizes, builds index
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple, Optional, Union

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import (
    COLAB_MODE,
    SLICES,
    MACBERTH_VECTORS_DIR,
    MACBERTH_SLICE_MODEL_DIR,
    EEBO_MODEL_NAME,
    MACBERTH_FINE_TUNED_DIR
)
from lib.eebo_sentences import stream_slice_sentences
from lib.eebo_id_map import EEBOIDMap

DEVICE: str
TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None

id_map = EEBOIDMap()
id_map.load()


def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_SLICE_MODEL_DIR / f"slice_{start}_{end}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def aligned_vectors_path(slice_id: str) -> Path:
    return MACBERTH_VECTORS_DIR / f"{slice_id}.npz"


def faiss_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_VECTORS_DIR / f"slice_{start}_{end}.faiss"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def vocab_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_VECTORS_DIR / f"slice_{start}_{end}.vocab"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_vectors(
    slice_id: str,
    embeddings: dict[str, list[np.ndarray]],
    doc_ids: dict[str, list[str]]
) -> None:
    path = aligned_vectors_path(slice_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat_tokens: list[str] = []
    flat_vectors: list[np.ndarray] = []
    flat_doc_ids: list[str] = []

    for token, vecs in embeddings.items():
        ids = doc_ids.get(token)
        if ids is None or len(ids) != len(vecs):
            raise ValueError(f"doc_ids mismatch for token {token}")

        for v, d in zip(vecs, ids, strict=True):
            flat_tokens.append(token)
            flat_vectors.append(v.astype(np.float32))
            flat_doc_ids.append(d)

    np.savez_compressed(
        path,
        tokens=np.array(flat_tokens, dtype=object),
        vectors=np.stack(flat_vectors),
        doc_ids=np.array(flat_doc_ids, dtype=object),
    )


def get_macberth_model(shared_only: bool = True) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        logger.info("Loading MacBERTh shared model...")
        tokenizer = AutoTokenizer.from_pretrained(EEBO_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(EEBO_MODEL_NAME)
        ft_dir = MACBERTH_FINE_TUNED_DIR
        if all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"]):
            logger.info("Loading fine-tuned shared weights...")
            state_dict = torch.load(ft_dir / "pytorch_model.bin", map_location="cpu")
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
            model.load_state_dict(state_dict, strict=False)
        model.eval()
        if shared_only:
            TOKENIZER, MODEL = tokenizer, model
    assert TOKENIZER is not None and MODEL is not None
    logger.info("Got MacBERTh shared")
    return TOKENIZER, MODEL


def _forward_batch(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, batch: list[str]) -> Tuple[np.ndarray, BatchEncoding]:
    batch_encoding = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    inputs = {k: v.to(DEVICE) for k, v in batch_encoding.items() if k != "offset_mapping"}
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1].cpu().numpy()
    return hidden_states, batch_encoding


def add_to_faiss_index(
    index: faiss.Index,
    vectors: np.ndarray,
    vector_ids: Optional[Union[np.ndarray, list[int]]] = None
) -> faiss.Index:
    """
    Add vectors to a FAISS index, optionally with explicit IDs.
    Returns the possibly wrapped index (IndexIDMap if needed).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if vector_ids is not None:
        ids = np.array(vector_ids, dtype=np.int64)
        if len(ids) != vectors.shape[0]:
            raise ValueError("Length of vector_ids must match number of vectors")
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        index.add_with_ids(vectors, ids)
    else:
        index.add(vectors)

    return index


def process_slice(
    slice_range: Tuple[int,int],
    force: bool = False,
    batch_size: int = 128,
    save_occurrence_vectors: bool = True
) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (force={force})")

    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)
    vectors_path = aligned_vectors_path(slice_id)

    tokenizer, shared_model = get_macberth_model(shared_only=True)
    slice_model_dir = slice_model_path(slice_range)
    if force or not any(slice_model_dir.iterdir()):
        logger.info(f"Saving per-slice MacBERTh model to {slice_model_dir}")
        shared_model.save_pretrained(slice_model_dir)
        tokenizer.save_pretrained(slice_model_dir)
        logger.info(f"Saved to {slice_model_dir}")

    model = AutoModelForMaskedLM.from_pretrained(slice_model_dir)
    model.to(DEVICE)
    model.eval()

    dim = model.config.hidden_size
    index: faiss.Index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    seen_words: set[str] = set()
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]] = defaultdict(list) if save_occurrence_vectors else None
    doc_ids_accum: Optional[DefaultDict[str, list[str]]] = defaultdict(list) if save_occurrence_vectors else None

    conn = get_connection()
    sentence_stream = stream_slice_sentences(conn, slice_range)

    if COLAB_MODE and DEVICE == "cuda":
        batch_size = min(batch_size, 64)

    batch: list[Tuple[str,str]] = []
    with torch.no_grad():
        for doc_id, sent in sentence_stream:
            batch.append((doc_id, sent))
            if len(batch) < batch_size:
                continue

            hidden_states, batch_encoding = _forward_batch(model, tokenizer, [s for _, s in batch])

            for b_idx, (doc_id, sent) in enumerate(batch):
                input_ids = batch_encoding["input_ids"][b_idx]
                offsets = batch_encoding["offset_mapping"][b_idx]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                current_word, current_vecs, last_end = "", [], None
                for idx, (_tok, (start, end)) in enumerate(zip(tokens, offsets, strict=True)):
                    if start == end:
                        continue
                    piece = sent[start:end]
                    if last_end is not None and start != last_end:
                        if current_word and current_vecs:
                            vec = np.mean(np.stack(current_vecs), axis=0).astype(np.float32)
                            vec /= max(np.linalg.norm(vec), 1e-12)
                            vector_id = id_map.get_numeric_id(
                                f"{slice_id}_{doc_id}_{start}_{end}"
                            )
                            index = add_to_faiss_index(index, vec.reshape(1, -1), [vector_id])
                            seen_words.add(current_word)

                            if save_occurrence_vectors and embeddings_accum is not None and doc_ids_accum is not None:
                                embeddings_accum[current_word].append(vec)
                                doc_ids_accum[current_word].append(doc_id)

                        current_word, current_vecs = "", []

                    current_word += piece
                    current_vecs.append(hidden_states[b_idx, idx])
                    last_end = end

                if current_word and current_vecs:
                    vec = np.mean(np.stack(current_vecs), axis=0).astype(np.float32)
                    vec /= max(np.linalg.norm(vec), 1e-12)
                    vector_id = id_map.get_numeric_id(f"{slice_id}_{current_word}_{doc_id}")
                    index = add_to_faiss_index(index, vec.reshape(1, -1), [vector_id])
                    seen_words.add(current_word)

                    if save_occurrence_vectors and embeddings_accum is not None and doc_ids_accum is not None:
                        embeddings_accum[current_word].append(vec)
                        doc_ids_accum[current_word].append(doc_id)

            batch.clear()

    conn.close()

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index at {index_path}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(seen_words)))
    logger.info(f"Saved vocab at {vocab_path}")

    if save_occurrence_vectors and embeddings_accum is not None and doc_ids_accum is not None:
        save_vectors(slice_id, embeddings_accum, doc_ids_accum)
        logger.info(f"Saved occurrence-level vectors at {vectors_path}")

    id_map.save()
    logger.info("Saved EEBO ID map")
    logger.info("Slice streaming & FAISS build complete.")


def build_all_slices(force: bool = False) -> None:
    for start, end in SLICES:
        process_slice((start, end), force=force, save_occurrence_vectors=True)


def main():
    global DEVICE

    if not COLAB_MODE:
        parser = argparse.ArgumentParser(description="Generate per-slice MacBERTh embeddings and FAISS indexes")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args()
        cli_force = args.force
    else:
        cli_force = False

    env_force = os.environ.get("EEBO_FORCE_OVERWRITE", "").lower()
    use_force = cli_force or env_force in {"1", "true", "yes"}

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Starting slice pipeline (force={use_force}, colab={COLAB_MODE})")
    build_all_slices(force=use_force)


if __name__ == "__main__":
    main()
