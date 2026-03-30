#!/usr/bin/env python

"""
mb_embedding_pipeline.py

Every vector in the system corresponds to exactly one token occurrence.
No vector represents an aggregate unless explicitly constructed outside the index.

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
from lib.slice_model_path import slice_model_path, vectors_path, faiss_slice_path
from lib.eebo_sentences import stream_slice_sentences
from lib.FaissIndex import FaissIndex as OccurrenceFaissIndex

SAVE_OCCURRENCE_VECTORS = os.getenv("SAVE_OCCURRENCE_VECTORS", "1") == "1"
DEVICE: str
TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None
_DEVICE: Optional[str] = None


@dataclass
class SentenceBatchItem:
    doc_id: str                             # pamphlet_tokens.doc_id
    sentence: str
    token_occurrence_ids: List[int]         # pamphlet_tokens.token_occurrence_id

@dataclass
class WordVector:
    word: str        # derived from tokenizer; not canonical (DB is source of truth)
    vector: np.ndarray
    vector_id: int  # pamphlet_tokens.token_occurrence_id
    doc_id: str     # pamphlet_tokens.doc_id



def get_device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def normalize_or_none(v: np.ndarray) -> Optional[np.ndarray]:
    """
    Normalize vector or return None if near-zero.

    This is the canonical entry point for all vectors entering FAISS space.
    """
    v = v.astype(np.float32, copy=False)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    return v / norm


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
    inputs = {k: v.to(get_device()) for k, v in batch_encoding.items() if k != "offset_mapping"}
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
    vec = normalize_or_none(np.mean(np.stack(current_vecs), axis=0))
    if vec is None:
        return None
    return WordVector(word=current_word, vector=vec, vector_id=vector_id, doc_id=doc_id)


def process_sentence(
    tokenizer: PreTrainedTokenizerBase,
    doc_id: str,
    sent: str,
    hidden_states: np.ndarray,
    batch_encoding: BatchEncoding,
    batch_index: int,
    token_occurrence_ids: list[int],
    index: OccurrenceFaissIndex,
    embeddings_accum: Optional[dict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[dict[str, list[str]]],
) -> None:
    """
    Process a single sentence: tokenize, compute word embeddings, add to FAISS index.
    Uses token_occurrence_id from DB for vector IDs (one per word).
    Aggregates subword embeddings per word.
    """

    # Extract tensors
    input_ids = cast(torch.Tensor, batch_encoding["input_ids"])
    offset_mapping = cast(torch.Tensor, batch_encoding["offset_mapping"])

    tokenizer_tokens = input_ids[batch_index].tolist()
    offsets = offset_mapping[batch_index].tolist()

    # if batch_index == 0:
    #     debug_sentence_alignment(
    #         sent,
    #         tokenizer_tokens,
    #         offsets,
    #         token_occurrence_ids,
    #         hidden_states,
    #         batch_index,
    #         tokenizer
    #     )

    current_word: str = ""
    current_vecs: list[np.ndarray] = []

    word_idx = 0  # index into token_occurrence_ids

    # Helper to flush a word occurrence to FAISS and accumulators
    def _add_word_to_faiss_and_accumulators(word: str, vecs: list[np.ndarray], vector_id: int) -> None:
        if not vecs:
            return
        wv = _flush_word(word, vecs, vector_id, doc_id)
        if not wv:
            return
        index.add(wv.vector.reshape(1, -1), [wv.vector_id])
        if SAVE_OCCURRENCE_VECTORS and embeddings_accum is not None and doc_ids_accum is not None:
            embeddings_accum[wv.word].append(wv.vector)
            doc_ids_accum[wv.word].append(wv.doc_id)

    for idx, (_tok, (start, end)) in enumerate(zip(tokenizer_tokens, offsets, strict=True)):
        if start == end:
            continue  # skip special tokens

        current_word += sent[start:end]
        current_vecs.append(hidden_states[batch_index, idx])

        # If next token starts a new word (gap in offsets), flush current word
        next_is_gap = (idx + 1 < len(offsets)) and (offsets[idx + 1][0] != end)
        if next_is_gap:
            if word_idx >= len(token_occurrence_ids):
                raise ValueError(
                    f"Word index {word_idx} exceeds token_occurrence_ids length {len(token_occurrence_ids)}"
                )
            _add_word_to_faiss_and_accumulators(current_word, current_vecs, token_occurrence_ids[word_idx])
            current_word, current_vecs = "", []
            word_idx += 1

    # Flush any remaining word at sentence end
    if current_word and current_vecs and word_idx < len(token_occurrence_ids):
        _add_word_to_faiss_and_accumulators(current_word, current_vecs, token_occurrence_ids[word_idx])


def process_batch(
    batch: list,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    index: OccurrenceFaissIndex,
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]],
    doc_ids_accum: Optional[DefaultDict[str, list[str]]],
) -> None:
    """
    Process a batch of sentences through the model and update FAISS index.
    """

    if not batch:
        return

    sentences = [item.sentence for item in batch]
    hidden_states, batch_encoding = _forward_batch(model, tokenizer, sentences)

    for batch_index, item in enumerate(batch):
        process_sentence(
            tokenizer=tokenizer,
            doc_id=item.doc_id,
            sent=item.sentence,
            hidden_states=hidden_states,
            batch_encoding=batch_encoding,
            batch_index=batch_index,
            token_occurrence_ids=item.token_occurrence_ids,
            index=index,
            embeddings_accum=embeddings_accum,
            doc_ids_accum=doc_ids_accum,
        )

    batch.clear()

    # Clean up
    if get_device() == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

def load_model_for_slice(start: int, end: int) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load the MacBERTh model for a given slice (start, end).

    Args:
        start: start year of slice
        end: end year of slice

    Returns:
        Tuple of (model, tokenizer) for the slice
    """
    slice_model_dir = MACBERTH_SLICE_MODEL_DIR / f"slice_{start}_{end}"

    if not slice_model_dir.exists():
        raise FileNotFoundError(f"Slice-specific model directory not found: {slice_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(slice_model_dir)
    model = AutoModelForMaskedLM.from_pretrained(slice_model_dir)
    logger.info(f"Loaded slice-specific MacBERTh model for {start}-{end}")

    model.to(get_device())
    model.eval()
    return model, tokenizer


def embed_word_with_model(
    word: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> np.ndarray:
    """
    Embed a single word using a provided model/tokenizer.

    Assumes model is already on the correct DEVICE and in eval() mode.

    Returns:
        L2-normalized embedding vector (1D np.ndarray)
    """
    inputs = tokenizer(
        [word],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(get_device()) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (1, seq_len, dim)

        attention_mask = inputs["attention_mask"]  # (1, seq_len)

        # zero out padding tokens
        masked_hidden = hidden * attention_mask.unsqueeze(-1)

        # mean over real tokens only
        token_counts = attention_mask.sum(dim=1, keepdim=True)
        vec = masked_hidden.sum(dim=1) / token_counts

        vec = vec.squeeze(0).cpu().numpy().astype(np.float32)

    normed = normalize_or_none(vec)
    if normed is None:
        raise ValueError(f"Zero or invalid embedding for word: '{word}'")
    return normed


def embed_word(word: str, start: int, end: int) -> np.ndarray:
    model, tokenizer = load_model_for_slice(start, end)
    return embed_word_with_model(word, model, tokenizer)


def embed_query(texts: list[str], start: int, end: int) -> np.ndarray:
    """
    Embed multiple words/phrases at once. Returns a 2D array (n_texts, dim).

    Args:
        texts: list of strings to embed
        start: slice start
        end: slice end

    Returns:
        2D numpy array of normalized embeddings
    """
    return np.stack([embed_word(t, start, end) for t in texts])


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
    model.to(get_device())
    model.eval()

    dim = model.config.hidden_size
    index = OccurrenceFaissIndex(dim)
    embeddings_accum: Optional[DefaultDict[str, list[np.ndarray]]] = defaultdict(list) if SAVE_OCCURRENCE_VECTORS else None
    doc_ids_accum: Optional[DefaultDict[str, list[str]]] = defaultdict(list) if SAVE_OCCURRENCE_VECTORS else None

    sentence_stream = stream_slice_sentences(conn, slice_range)

    if COLAB_MODE and get_device() == "cuda":
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
                process_batch(batch, model, tokenizer, index, embeddings_accum, doc_ids_accum )
                if processed_count % log_every == 0:
                    logger.info(f"Processed {processed_count} sentences")

        # process any remaining
        if batch:
            process_batch(batch, model, tokenizer, index, embeddings_accum, doc_ids_accum )
            logger.info(f"Processed {processed_count} sentences (final)")

    index.save(str(index_path))
    logger.info(f"Saved FAISS index at {index_path}")

    if SAVE_OCCURRENCE_VECTORS and embeddings_accum is not None and doc_ids_accum is not None:
        save_vectors(slice_id, embeddings_accum, doc_ids_accum)

    logger.info("Slice streaming & FAISS build complete.")

    del model, tokenizer, index, embeddings_accum, doc_ids_accum
    gc.collect()
    if get_device() == "cuda":
        torch.cuda.empty_cache()


def build_all_slices() -> None:
    conn = get_connection()
    for start, end in SLICES:
        process_slice(conn, (start, end))
    conn.close()



def debug_sentence_alignment(
    sent: str,
    tokenizer_tokens: list[int],
    offsets: list[tuple[int, int]],
    token_occurrence_ids: list[int],
    hidden_states: np.ndarray,
    batch_index: int,
    tokenizer: PreTrainedTokenizerBase
) -> None:
    print("\n=== DEBUG SENTENCE ===")
    print(f"TEXT: {sent}")
    print(f"len(token_occurrence_ids): {len(token_occurrence_ids)}")
    print(f"len(tokenizer_tokens): {len(tokenizer_tokens)}")
    print()

    decoded_tokens = tokenizer.convert_ids_to_tokens(tokenizer_tokens)

    for idx, (tok, (start, end)) in enumerate(zip(decoded_tokens, offsets, strict=True)):
        piece = sent[start:end] if start != end else ""
        occ_id = token_occurrence_ids[idx] if idx < len(token_occurrence_ids) else None

        print(f"{idx:03d} | tok={tok:>12} | span=({start:>3},{end:>3}) | text='{piece}' | occ_id={occ_id}")


def main():
    logger.info(f"Using device: {get_device()}")

    logger.info(f"Starting slice pipeline (colab={COLAB_MODE})")
    build_all_slices()


if __name__ == "__main__":
    main()
