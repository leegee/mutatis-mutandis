#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (MacBERTh per-slice models) and build FAISS indexes.
- MacBERTh: embeddings per slice
- FAISS: always loads saved embeddings, flattens, normalizes, builds index

This version drops fastText
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple, Optional, Union, Callable, cast

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizerBase, PreTrainedModel, BatchEncoding

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import (
    COLAB_MODE,
    SLICES,
    MACBERTH_ALIGNED_VECTORS_DIR,
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
    return MACBERTH_ALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def faiss_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_ALIGNED_VECTORS_DIR / f"slice_{start}_{end}.faiss"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def vocab_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_ALIGNED_VECTORS_DIR / f"slice_{start}_{end}.vocab"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def _save_vectors(path: Path, embeddings: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokens = list(embeddings.keys())
    vectors = np.stack(list(embeddings.values())).astype(np.float32)
    np.savez_compressed(path, tokens=tokens, vectors=vectors)


def _load_vectors(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Vectors missing: {path}")
    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"]
    vectors = data["vectors"]
    return {tok: vectors[i] for i, tok in enumerate(tokens)}


def save_aligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    _save_vectors(aligned_vectors_path(slice_id), embeddings)


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    return _load_vectors(aligned_vectors_path(slice_id))



def has_fine_tuned_weights(ft_dir: Path) -> bool:
    return all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"])


def get_macberth_model(shared_only: bool = True) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load shared MacBERTh model (or per-slice later)"""
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        logger.info("Loading MacBERTh shared model...")
        tokenizer = AutoTokenizer.from_pretrained(EEBO_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(EEBO_MODEL_NAME)
        if has_fine_tuned_weights(MACBERTH_FINE_TUNED_DIR):
            logger.info("Loading fine-tuned shared weights...")
            state_dict = torch.load(MACBERTH_FINE_TUNED_DIR / "pytorch_model.bin", map_location="cpu")
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


def _accumulate_tokens(
    tokenizer: PreTrainedTokenizerBase,
    batch_encoding,
    hidden_states: np.ndarray,
    embeddings_accum: DefaultDict[str, list[np.ndarray]],
    doc_ids_accum: DefaultDict[str, list[str]],
    sentences: list[tuple[str, str]]
) -> None:
    input_ids = batch_encoding["input_ids"]
    offsets = batch_encoding["offset_mapping"]

    for b_idx, (doc_id, sent) in enumerate(sentences):
        token_ids = input_ids[b_idx].tolist()
        token_offsets = offsets[b_idx].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        if len(tokens) != len(token_offsets):
            raise ValueError(f"Mismatch: {len(tokens)} tokens vs {len(token_offsets)} offsets")

        current_word, current_vecs, last_end = "", [], None

        for idx, (_tok, (start, end)) in enumerate(zip(tokens, token_offsets, strict=True)):
            if start == end:
                continue
            piece = sent[start:end]
            if last_end is not None and start != last_end:
                if current_word and current_vecs:
                    vec = np.mean(np.stack(current_vecs), axis=0)
                    embeddings_accum[current_word].append(vec)
                    doc_ids_accum[current_word].append(doc_id)
                current_word, current_vecs = "", []
            current_word += piece
            current_vecs.append(hidden_states[b_idx, idx])
            last_end = end

        if current_word and current_vecs:
            vec = np.mean(np.stack(current_vecs), axis=0)
            embeddings_accum[current_word].append(vec)
            doc_ids_accum[current_word].append(doc_id)



def generate_embeddings_per_slice(
    slice_range: Tuple[int,int],
    force: bool = False,
    batch_size: int = 128  # default
) -> Tuple[Dict[str,np.ndarray], DefaultDict[str,List[str]]]:

    if COLAB_MODE and DEVICE == "cuda":
        batch_size = min(batch_size, 64)

    embeddings_accum: DefaultDict[str, list[np.ndarray]] = defaultdict(list)
    doc_ids_accum: DefaultDict[str, list[str]] = defaultdict(list)

    tokenizer, shared_model = get_macberth_model(shared_only=True)

    # Copy shared model to per-slice folder (for semantic drift tracking)
    slice_model_dir = slice_model_path(slice_range)
    if force or not any(slice_model_dir.iterdir()):
        logger.info(f"Saving per-slice MacBERTh model to {slice_model_dir}")
        shared_model.save_pretrained(slice_model_dir)
        tokenizer.save_pretrained(slice_model_dir)
        logger.info(f"Saved to {slice_model_dir}")

    model = AutoModelForMaskedLM.from_pretrained(slice_model_dir)
    model.to(DEVICE)
    model.eval()

    logger.info("Connecting to DB")
    conn = get_connection()

    logger.info("Connected to DB, streaming sentences")
    sentence_stream = stream_slice_sentences(conn, slice_range)
    batch: list[tuple[str,str]] = []
    sentence_count = 0

    with torch.no_grad():
        for sent_tuple in sentence_stream:
            batch.append(sent_tuple)
            if len(batch) < batch_size:
                continue
            hidden_states, batch_encoding = _forward_batch(model, tokenizer, [s for _, s in batch])
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, doc_ids_accum, batch)
            sentence_count += len(batch)
            if sentence_count % 100 == 0:
                logger.info("Processed %d sentences", sentence_count)
            batch.clear()

        if batch:
            hidden_states, batch_encoding = _forward_batch(model, tokenizer, [s for _, s in batch])
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, doc_ids_accum, batch)
            sentence_count += len(batch)

    conn.close()
    logger.info("Total sentences processed: %d", sentence_count)
    logger.info("Averaging embeddings for %d tokens", len(embeddings_accum))

    final_embeddings: Dict[str,np.ndarray] = {}
    for token, vecs in embeddings_accum.items():
        final_embeddings[token] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    return final_embeddings, doc_ids_accum



def add_to_faiss_index(
    index: faiss.Index,
    vectors: np.ndarray,
    vector_ids: Optional[Union[np.ndarray, list[int]]] = None
) -> None:
    """
    Add vectors to a FAISS index, optionally with explicit IDs (e.g., doc_id).
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    if vector_ids is not None:
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        ids = np.array(vector_ids, dtype=np.int64)
        if len(ids) != vectors.shape[0]:
            raise ValueError("Length of vector_ids must match number of vectors")
        add_with_ids_fn = cast(Callable[[np.ndarray, np.ndarray], None], index.add_with_ids)
        add_with_ids_fn(vectors, ids)
    else:
        add_fn = cast(Callable[[np.ndarray], None], index.add)
        add_fn(vectors)



def build_index_for_slice(slice_range: Tuple[int,int], force: bool = False) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (force={force})")

    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)
    vectors_path = aligned_vectors_path(slice_id)

    if force or not vectors_path.exists():
        embeddings, doc_ids_accum = generate_embeddings_per_slice(slice_range, force)
        save_aligned_vectors(slice_id, embeddings)
    else:
        embeddings = load_aligned_vectors(slice_id)
        doc_ids_accum = defaultdict(list)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings for slice {slice_id}, skipping FAISS build")
        return

    all_vectors, all_ids = [], []
    for word in words:
        vecs_list = embeddings[word] if isinstance(embeddings[word], list) else [embeddings[word]]
        doc_ids_list = doc_ids_accum[word] if word in doc_ids_accum else list(range(len(vecs_list)))
        if len(vecs_list) != len(doc_ids_list):
            doc_ids_list = list(range(len(vecs_list)))
        for vec, doc_id in zip(vecs_list, doc_ids_list, strict=True):
            all_vectors.append(vec)
            all_ids.append(id_map.get_numeric_id(doc_id))

    vectors = np.stack(all_vectors).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vector_ids = np.array(all_ids, dtype=np.int64)

    dim = vectors.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    add_to_faiss_index(index, vectors, vector_ids)

    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index at {index_path}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logger.info(f"Saved vocab at {vocab_path}")

    id_map.save()
    logger.info("Saved EEBO ID map")
    logger.info("FAISS build complete.")



def build_all_slices(force: bool = False) -> None:
    for start, end in SLICES:
        build_index_for_slice((start, end), force)



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
