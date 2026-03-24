#!/usr/bin/env python
"""
mb_embedding_pipeline.py

Generate token embeddings per slice (MacBERTh per-slice models) and build FAISS indexes.
- MacBERTh: embeddings per slice
- FAISS: always loads saved embeddings, flattens, normalizes, builds index
"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple, Optional,  cast, Any, Mapping
from psycopg import Connection
from dataclasses import dataclass
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
from lib.FaissIndex import FaissIndex

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

def vectors_path(slice_id: str) -> Path:
    return MACBERTH_VECTORS_DIR / f"{slice_id}.npz"

def vocab_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_VECTORS_DIR / f"slice_{start}_{end}.vocab"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def faiss_slice_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_VECTORS_DIR / f"slice_{start}_{end}.faiss"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_vectors(slice_id: str) -> dict[str, list[np.ndarray]]:
    """
    Load occurrence-level vectors saved with save_vectors.
    Returns dict mapping token -> list of vectors (all occurrences).
    """
    path = vectors_path(slice_id)
    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"]
    vectors = data["vectors"]

    result: dict[str, list[np.ndarray]] = {}
    for token, vec in zip(tokens, vectors, strict=True):
        result.setdefault(token, []).append(vec.astype(np.float32))

    return result

def save_vectors(
    slice_id: str,
    embeddings: dict[str, list[np.ndarray]],
    doc_ids: dict[str, list[str]]
) -> None:
    path = vectors_path(slice_id)
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


@dataclass
class WordVector:
    word: str
    vector: np.ndarray
    vector_id: int


def _flush_word(
    current_word: str,
    current_vecs: list[np.ndarray],
    slice_id: str,
    doc_id: str,
    word_start: int,
    last_end: int
) -> Optional[WordVector]:
    """Compute the normalized vector for a word, return WordVector or None if invalid."""
    if not current_word or not current_vecs or word_start is None or last_end is None:
        return None

    vec = np.mean(np.stack(current_vecs), axis=0).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return None
    vec /= norm

    vector_id = id_map.get_numeric_id(f"{slice_id}_{doc_id}_{word_start}_{last_end}")

    return WordVector(word=current_word, vector=vec, vector_id=vector_id)


def process_sentence(
    sent: str,
    hidden_states: np.ndarray,
    batch_encoding: BatchEncoding,
    b_idx: int,
    slice_id: str,
    doc_id: str,
    index: FaissIndex,
    seen_words: set[str],
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[DefaultDict[str, list[str]]],
    save_occurrence_vectors: bool
) -> None:
    enc = cast(Mapping[str, Any], batch_encoding)

    input_ids = enc["input_ids"][b_idx]
    offsets = enc["offset_mapping"][b_idx]

    assert TOKENIZER is not None
    tokenizer_tokens = TOKENIZER.convert_ids_to_tokens(input_ids.tolist())

    current_word: str = ""
    current_vecs: list[np.ndarray] = []
    last_end: Optional[int] = None
    word_start: Optional[int] = None

    for idx, (_tok, (start, end)) in enumerate(zip(tokenizer_tokens, offsets, strict=True)):
        if start == end:
            continue

        if word_start is None:
            word_start = start

        piece = sent[start:end]

        # flush previous word if there is a gap
        if last_end is not None and start != last_end and word_start is not None:
            wv = _flush_word(current_word, current_vecs, slice_id, doc_id, word_start, last_end)
            if wv:
                index.add(wv.vector.reshape(1, -1), [wv.vector_id])
                seen_words.add(wv.word)
                if save_occurrence_vectors and embeddings_accum is not None and doc_ids_accum is not None:
                    embeddings_accum[wv.word].append(wv.vector)
                    doc_ids_accum[wv.word].append(doc_id)
            current_word, current_vecs = "", []
            word_start = start  # start new word at current piece

        current_word += piece
        current_vecs.append(hidden_states[b_idx, idx])
        last_end = end

    # flush any leftover word at end of sentence
    if word_start is not None and last_end is not None:
        wv = _flush_word(current_word, current_vecs, slice_id, doc_id, word_start, last_end)
        if wv:
            index.add(wv.vector.reshape(1, -1), [wv.vector_id])
            seen_words.add(wv.word)
            if save_occurrence_vectors and embeddings_accum is not None and doc_ids_accum is not None:
                embeddings_accum[wv.word].append(wv.vector)
                doc_ids_accum[wv.word].append(doc_id)


def process_batch(
    batch: list[Tuple[str,str]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    slice_id: str,
    index: FaissIndex,
    seen_words: set[str],
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[DefaultDict[str, list[str]]],
    save_occurrence_vectors: bool
) -> None:
    """Process a batch of sentences through the model and update FAISS index."""
    if not batch:
        return

    hidden_states, batch_encoding = _forward_batch(model, tokenizer, [s for _, s in batch])
    for b_idx, (doc_id, sent) in enumerate(batch):
        process_sentence(
            sent=sent,
            hidden_states=hidden_states,
            batch_encoding=batch_encoding,
            b_idx=b_idx,
            slice_id=slice_id,
            doc_id=doc_id,
            index=index,
            seen_words=seen_words,
            embeddings_accum=embeddings_accum,
            doc_ids_accum=doc_ids_accum,
            save_occurrence_vectors=save_occurrence_vectors
        )
    batch.clear()


def process_slice(
    conn: Connection,
    slice_range: Tuple[int,int],
    batch_size: int = 128,
    save_occurrence_vectors: bool = True
) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id}")

    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)
    vectors_path = vectors_path(slice_id)

    tokenizer, shared_model = get_macberth_model(shared_only=True)
    slice_model_dir = slice_model_path(slice_range)

    logger.info(f"Saving per-slice MacBERTh model to {slice_model_dir}")
    shared_model.save_pretrained(slice_model_dir)
    tokenizer.save_pretrained(slice_model_dir)
    logger.info(f"Saved to {slice_model_dir}")

    model = AutoModelForMaskedLM.from_pretrained(slice_model_dir)
    model.to(DEVICE)
    model.eval()

    dim = model.config.hidden_size
    index = FaissIndex(dim)

    seen_words: set[str] = set()
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]] = defaultdict(list) if save_occurrence_vectors else None
    doc_ids_accum: Optional[DefaultDict[str, list[str]]] = defaultdict(list) if save_occurrence_vectors else None

    sentence_stream = stream_slice_sentences(conn, slice_range)

    if COLAB_MODE and DEVICE == "cuda":
        batch_size = min(batch_size, 64)

    batch: list[Tuple[str,str]] = []
    with torch.no_grad():
        for doc_id, sent in sentence_stream:
            batch.append((doc_id, sent))
            if len(batch) >= batch_size:
                process_batch(batch, model, tokenizer, slice_id, index, seen_words, embeddings_accum, doc_ids_accum, save_occurrence_vectors)

        # process any leftover sentences
        process_batch(batch, model, tokenizer, slice_id, index, seen_words, embeddings_accum, doc_ids_accum, save_occurrence_vectors)

    index.save(str(index_path))
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


def build_all_slices() -> None:
    conn = get_connection()
    for start, end in SLICES:
        process_slice(conn, (start, end), save_occurrence_vectors=True)
    conn.close()


def main():
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Starting slice pipeline (colab={COLAB_MODE})")
    build_all_slices()


if __name__ == "__main__":
    main()
