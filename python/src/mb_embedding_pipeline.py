#!/usr/bin/env python
"""
mb_embedding_pipeline.py

Pipeline for generating contextual word embeddings and FAISS indexes from EEBO
Early Modern English text slices using MacBERTh models.

This module supports both occurrence-level and token-level representations:

- Occurrence-level: every word occurrence is embedded and indexed with its
  token_occurrence_id, enabling full traceability back to the database.
- Token-level: mean embeddings are computed per token across all occurrences
  within a slice, enabling semantic similarity and drift analysis.

This module provides functions to:

1. Stream tokenized sentences and aligned token_occurrence_ids from the database.
2. Compute normalized word embeddings from MacBERTh hidden states.
3. Build occurrence-level FAISS indexes keyed by token_occurrence_id.
4. Accumulate per-token embeddings and compute mean vectors.
5. Build token-level FAISS indexes (no IDs; index position defines token identity).
6. Optionally persist occurrence-level embeddings for reuse.
7. Handle batch processing and memory management for large corpora.
8. Run efficiently on CPU or GPU (including Colab environments).

Key Concepts:

- WordVector:
  Represents a single word occurrence embedding, including:
    - word string
    - normalized vector
    - token_occurrence_id (DB primary key)
    - doc_id

- SentenceBatchItem:
  Represents a unit of processing:
    - doc_id
    - raw sentence text
    - aligned token_occurrence_ids

- Occurrence-level FAISS index:
  Stores one vector per token occurrence.
  IDs correspond to token_occurrence_id, enabling direct lookup in the database.
  This index supports traceability and retrieval of exact textual contexts.

- Token-level FAISS index:
  Stores one mean vector per token (per slice), computed across all occurrences.
  No explicit IDs are stored; index position corresponds to token ordering.
  This index supports semantic similarity, clustering, and drift analysis.

Design Notes:

- The database is the authoritative source for token metadata.
  Occurrence-level FAISS results are resolved via token_occurrence_id-to-DB lookup.

- The system separates two analytical layers:
    - Occurrence-level (evidence, context, traceability)
    - Token-level (abstraction, semantics, drift)

- Token-level FAISS requires an external mapping (e.g. ordered token list)
  if reconstruction of token strings from index positions is needed.

- Occurrence-level vector persistence is optional and intended for:
    - offline analysis
    - reproducibility
    - rebuilding indexes without recomputation

Workflow:

1. Load or initialize the shared MacBERTh model (optionally with fine-tuned weights).
2. Stream sentences and token_occurrence_ids from the database for a slice.
3. Tokenize each sentence and compute contextual embeddings.
4. Aggregate subword embeddings into word-level vectors.
5. Normalize vectors and:
    - add to occurrence-level FAISS (with token_occurrence_id)
    - accumulate per-token vectors for later averaging
6. After the slice:
    - save occurrence-level FAISS index
    - compute mean vectors per token
    - build and save token-level FAISS index
7. Optionally persist occurrence-level vectors for reuse.

"""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Tuple, Optional, List, cast
import os
import gc
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
from lib.FaissIndex import FaissIndex
from lib.TokenFaissIndex import TokenFaissIndex

SAVE_OCCURRENCE_VECTORS = os.getenv("SAVE_OCCURRENCE_VECTORS", "0") == "1"
DEVICE: str
TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None


@dataclass
class SentenceBatchItem:
    doc_id: str                             # pamphlet_tokens.doc_id
    sentence: str
    token_occurrence_ids: List[int]         # pamphlet_tokens.token_occurrence_id

@dataclass
class WordVector:
    word: str
    vector: np.ndarray
    vector_id: int # pamphlet_tokens.token_occurrence_id
    doc_id: str    # pamphlet_tokens.doc_id

def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    path = MACBERTH_SLICE_MODEL_DIR / f"slice_{start}_{end}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def vectors_path(slice_id: str) -> Path:
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
    logger.info(f"Saved occurrence-level vectors at {path}")


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


def _forward_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: list[str]
) -> Tuple[np.ndarray, BatchEncoding]:
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


def _flush_word(
    current_word: str,
    current_vecs: List[np.ndarray],
    vector_id: int,
    doc_id: str,
) -> Optional[WordVector]:
    """
    Compute the normalized vector for a word occurrence and return a WordVector.
    vector_id and doc_id are DB token_occurrence_id and doc_id.
    """
    if not current_word or not current_vecs:
        return None

    # Compute the mean vector across all subword embeddings for this word
    vec = np.mean(np.stack(current_vecs), axis=0).astype(np.float32)

    # Compute vector norm for normalization
    norm = np.linalg.norm(vec)

    # Skip if the vector is effectively zero to avoid invalid embeddings
    if norm < 1e-12:
        return None

    # Normalize vector to unit length (L2 normalization)
    vec /= norm

    return WordVector(word=current_word, vector=vec, vector_id=vector_id, doc_id=doc_id)


def build_token_level_index(
    token_vectors_accum: DefaultDict[str, List[np.ndarray]],
    slice_range: tuple[int, int]
) -> None:
    """
    Compute mean vectors per token from accumulated occurrence vectors
    and build a token-level FAISS index for semantic search.

    Args:
        token_vectors_accum: Dict mapping token -> list of occurrence vectors
        slice_range: The slice being processed (start, end)
    """
    if not token_vectors_accum:
        logger.warning("No token vectors accumulated; skipping token-level FAISS build")
        return

    # Sort tokens for stable ordering
    tokens_ordered = sorted(token_vectors_accum.keys())
    mean_vectors = []

    for token in tokens_ordered:
        vecs = token_vectors_accum[token]
        mean_vec = np.mean(np.stack(vecs), axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm < 1e-12:
            continue
        mean_vec /= norm
        mean_vectors.append(mean_vec)

    if not mean_vectors:
        logger.warning("No valid mean vectors found; skipping token-level FAISS build")
        return

    mean_vectors_np = np.stack(mean_vectors)

    # Build TokenFaissIndex
    token_index = TokenFaissIndex(mean_vectors_np.shape[1])
    token_index.add(mean_vectors_np)

    # Save
    token_index_path = MACBERTH_VECTORS_DIR / f"slice_{slice_range[0]}_{slice_range[1]}.token.faiss"
    token_index.save(str(token_index_path))
    logger.info(f"Saved token-level FAISS index at {token_index_path}")


def process_sentence(
    doc_id: str,
    sent: str,
    hidden_states: np.ndarray,
    batch_encoding: BatchEncoding,
    batch_index: int,
    token_occurrence_ids: list[int],
    index: FaissIndex,
    embeddings_accum: Optional[dict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[dict[str, list[str]]],
    token_vectors_accum: DefaultDict[str, List[np.ndarray]],
) -> None:
    """
    Process a single sentence: tokenize, compute word embeddings, add to FAISS index.
    Uses token_occurrence_id from DB for vector IDs.
    """

    # Extract tensors explicitly for type checker
    input_ids = cast(torch.Tensor, batch_encoding["input_ids"])
    offset_mapping = cast(torch.Tensor, batch_encoding["offset_mapping"])

    tokenizer_tokens = input_ids[batch_index].tolist()
    offsets = offset_mapping[batch_index].tolist()  # If offsets are a tensor; else leave as-is

    current_word: str = ""
    current_vecs: list[np.ndarray] = []
    current_ids: list[int] = []

    # helper to flush a word occurrence
    def _handle_word_flush(
        word: str,
        vecs: list[np.ndarray],
        vector_id: int
    ) -> None:
        wv = _flush_word(word, vecs, vector_id, doc_id)
        if not wv:
            return
        index.add(wv.vector.reshape(1, -1), [wv.vector_id])
        if SAVE_OCCURRENCE_VECTORS and embeddings_accum is not None and doc_ids_accum is not None:
            embeddings_accum[wv.word].append(wv.vector)
            doc_ids_accum[wv.word].append(wv.doc_id)
        # Add to token-level accumulator for mean
        token_vectors_accum[word].append(wv.vector)


    for idx, (_tok, (start, end)) in enumerate(zip(tokenizer_tokens, offsets, strict=True)):
        if start == end:
            continue

        piece = sent[start:end]
        current_word += piece
        current_vecs.append(hidden_states[batch_index, idx])

        # Append only if token_occurrence_ids has this idx
        if idx < len(token_occurrence_ids):
            current_ids.append(token_occurrence_ids[idx])

        # Flush if next token is a gap
        if idx + 1 < len(offsets) and offsets[idx + 1][0] != end:
            if not current_ids:
                return  # skip this word entirely
            vector_id = current_ids[0]
            _handle_word_flush(current_word, current_vecs, vector_id)
            current_word, current_vecs, current_ids = "", [], []

    # Flush any leftover word at sentence end
    if current_word and current_ids:
        _handle_word_flush(current_word, current_vecs, current_ids[0])


def process_batch(
    batch: list[SentenceBatchItem],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    index: FaissIndex,
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[DefaultDict[str, list[str]]],
    token_vectors_accum: DefaultDict[str, List[np.ndarray]]
) -> None:
    """
    Process a batch of sentences through the model and update FAISS index.
    Each batch element is (doc_id, sentence, token_occurrence_ids)
    """
    if not batch:
        return

    sentences = [item.sentence for item in batch]
    hidden_states, batch_encoding = _forward_batch(model, tokenizer, sentences)

    for batch_index, item in enumerate(batch):
        process_sentence(
            doc_id=item.doc_id,
            sent=item.sentence,
            hidden_states=hidden_states,
            batch_encoding=batch_encoding,
            batch_index=batch_index,
            token_occurrence_ids=item.token_occurrence_ids,
            index=index,
            embeddings_accum=embeddings_accum,
            doc_ids_accum=doc_ids_accum,
            token_vectors_accum=token_vectors_accum
        )

    batch.clear()

    # Clean up memory
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def process_slice(
    conn: Connection,
    slice_range: tuple[int, int],
    batch_size: int = 128,
) -> None:
    """
    Process a slice of documents: generate word embeddings and build FAISS index.
    """
    log_every = 1000
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    index_path = faiss_slice_path(slice_range)

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
    token_vectors_accum: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]] = defaultdict(list) if SAVE_OCCURRENCE_VECTORS else None
    doc_ids_accum: Optional[DefaultDict[str, list[str]]] = defaultdict(list) if SAVE_OCCURRENCE_VECTORS else None

    sentence_stream = stream_slice_sentences(conn, slice_range)

    if COLAB_MODE and DEVICE == "cuda":
        batch_size = min(batch_size, 32)

    batch: list[SentenceBatchItem] = []
    processed_count = 0

    with torch.no_grad():
        for doc_id, sent, token_occurrence_ids in sentence_stream:
            batch.append(SentenceBatchItem(
                doc_id=doc_id,
                sentence=sent,
                token_occurrence_ids=token_occurrence_ids
            ))
            processed_count += 1

            if len(batch) >= batch_size:
                process_batch(batch, model, tokenizer, index, embeddings_accum, doc_ids_accum, token_vectors_accum  )
                if processed_count % log_every == 0:
                    logger.info(f"Processed {processed_count} sentences")

        # process any remaining
        if batch:
            process_batch(batch, model, tokenizer, index, embeddings_accum, doc_ids_accum, token_vectors_accum  )
            logger.info(f"Processed {processed_count} sentences (final)")

    index.save(str(index_path))
    logger.info(f"Saved FAISS index at {index_path}")

    build_token_level_index(token_vectors_accum, slice_range)

    if SAVE_OCCURRENCE_VECTORS and embeddings_accum is not None and doc_ids_accum is not None:
        save_vectors(slice_id, embeddings_accum, doc_ids_accum)

    logger.info("Slice streaming & FAISS build complete.")

    del model, tokenizer, index, embeddings_accum, doc_ids_accum
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def build_all_slices() -> None:
    conn = get_connection()
    for start, end in SLICES:
        process_slice(conn, (start, end))
    conn.close()


def main():
    global DEVICE
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")

    logger.info(f"Starting slice pipeline (colab={COLAB_MODE})")
    build_all_slices()


if __name__ == "__main__":
    main()
