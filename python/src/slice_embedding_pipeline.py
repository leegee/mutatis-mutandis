#!/usr/bin/env python
"""
slice_embedding_pipeline.py

Generate token embeddings per slice (aligned or unaligned) and build FAISS indexes.

- Embeddings: train or load fastText models per slice, or compute aligned embeddings
- FAISS indices: stored per slice in separate subdirs for aligned vs unaligned
- Provides methods to add and search the index
- Env var `USE_ALIGNED_FASTTEXT_VECTORS` and flag `--aligned`

    Corpus (slice text file: SLICES_DIR/1625-1629.txt)
            │
            V
    [Train fastText model once]
            │
            V
    Slice fastText model (slice_1625_1629.bin)
    * Stored in FASTTEXT_SLICE_MODEL_DIR
    * Trained once per slice
    * Can be reused for:
        - Unaligned embeddings
        - Aligned embeddings
            │
            V
    Generate Unaligned Vectors
    (raw embeddings from fastText)
            │
            V
    Unaligned vectors (slice_1625_1629.npz)
    * Stored in FASTTEXT_UNALIGNED_VECTORS_DIR
    * Used to build:
        - FAISS index (unaligned)
            │
            V
    FAISS index (slice_1625_1629.faiss, unaligned)
    * Stored in FASTTEXT_UNALIGNED_VECTORS_DIR
    * Queries raw vectors
    * Vocabulary mapping stored in slice_1625_1629.vocab
            │
            V
    Align Vectors to Reference Slice
    * Takes unaligned vectors
    * Applies orthogonal Procrustes rotation
    * Preserves terms (vocab), only vector coordinates change
            │
            V
    Aligned vectors (slice_1625_1629.npz)
    * Stored in FASTTEXT_ALIGNED_VECTORS_DIR
            │
            V
    FAISS index (slice_1625_1629.faiss, aligned)
    * Stored in FASTTEXT_ALIGNED_VECTORS_DIR
    * Queries aligned vectors
    * Vocabulary mapping shared with unaligned (terms unchanged)

"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Callable, cast, DefaultDict, List, Dict, Tuple, Optional
from collections import defaultdict

import fasttext
import faiss
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    PreTrainedTokenizerBase,
    PreTrainedModel
)

from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.eebo_config import (
    SLICES,
    SLICES_DIR,
    FASTTEXT_PARAMS,
    FASTTEXT_SLICE_MODEL_DIR,
    MACBERTH_SLICE_MODEL_DIR,
    MACBERTH_ALIGNED_VECTORS_DIR,
    FASTTEXT_UNALIGNED_VECTORS_DIR,
    FASTTEXT_ALIGNED_VECTORS_DIR,
    EEBO_MODEL_NAME,
    MACBERTH_FINE_TUNED_DIR
)
from lib.eebo_anchor_builder import get_anchors
from lib.eebo_sentences import stream_slice_sentences


# Cache
TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None

# Paths
def unaligned_vectors_path(slice_id: str, backend: str) -> Path:
    if backend.lower() == "macberth":
        base = MACBERTH_ALIGNED_VECTORS_DIR
    else:
        base = FASTTEXT_UNALIGNED_VECTORS_DIR
    return base / f"{slice_id}.npz"


def aligned_vectors_path(slice_id: str, backend: str) -> Path:
    if backend.lower() == "macberth":
        base = MACBERTH_ALIGNED_VECTORS_DIR
    else:
        base = FASTTEXT_ALIGNED_VECTORS_DIR
    return base / f"{slice_id}.npz"


def slice_model_path(slice_range: tuple[int,int], backend: str) -> Path:
    start, end = slice_range
    if backend.lower() == "macberth":
        base = MACBERTH_SLICE_MODEL_DIR
    else:
        base = FASTTEXT_SLICE_MODEL_DIR
    return base / f"slice_{start}_{end}.bin"



def load_macberth_vectors(slice_id: str) -> dict[str, np.ndarray]:
    """
    Load MacBERTh embeddings for a given slice (new 2D format).

    Returns:
        dict[token -> vector]
    """
    path = aligned_vectors_path(slice_id, "macberth")

    if not path.exists():
        raise FileNotFoundError(f"MacBERTh vectors not found: {path}")

    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"]
    vectors = data["vectors"]

    if len(tokens) != len(vectors):
        raise ValueError(f"Mismatch in tokens vs vectors: {len(tokens)} vs {len(vectors)}")

    return {tok: vectors[i] for i, tok in enumerate(tokens)}


def faiss_slice_path(slice_range: tuple[int,int], aligned: bool, backend: str) -> Path:
    """Return the path to the FAISS index for a slice, backend-aware."""
    backend = backend.lower()
    if backend == "fasttext":
        base_dir = FASTTEXT_ALIGNED_VECTORS_DIR if aligned else FASTTEXT_UNALIGNED_VECTORS_DIR
    elif backend == "macberth":
        base_dir = MACBERTH_ALIGNED_VECTORS_DIR
    else:
        raise NotImplementedError(f"Unknown backend: {backend}")

    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.faiss"


def vocab_slice_path(slice_range: tuple[int,int], aligned: bool, backend: str) -> Path:
    """Return the path to the vocabulary file for a slice, backend-aware."""
    backend = backend.lower()
    if backend == "fasttext":
        base_dir = FASTTEXT_ALIGNED_VECTORS_DIR if aligned else FASTTEXT_UNALIGNED_VECTORS_DIR
    elif backend == "macberth":
        base_dir = MACBERTH_ALIGNED_VECTORS_DIR
    else:
        raise ValueError(f"Unknown backend: {backend}")

    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.vocab"


# MB Model loader
def get_macberth_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    global TOKENIZER, MODEL

    if TOKENIZER is None or MODEL is None:
        logger.info("Loading MacBERTh model...")
        tokenizer = AutoTokenizer.from_pretrained(EEBO_MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(EEBO_MODEL_NAME)

        if has_fine_tuned_weights(MACBERTH_FINE_TUNED_DIR):
            logger.info("Loading fine-tuned MacBERTh weights...")
            state_dict = torch.load(
                MACBERTH_FINE_TUNED_DIR / "pytorch_model.bin",
                map_location="cpu"
            )
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        TOKENIZER = tokenizer
        MODEL = model

    assert TOKENIZER is not None and MODEL is not None
    return TOKENIZER, MODEL


# Internal IO

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


# Public IO

def save_aligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray], backend: str) -> None:
    _save_vectors(aligned_vectors_path(slice_id, backend), embeddings)


def load_aligned_vectors(slice_id: str, backend: str) -> dict[str, np.ndarray]:
    return _load_vectors(aligned_vectors_path(slice_id, backend))


def save_unaligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray], backend: str) -> None:
    _save_vectors(unaligned_vectors_path(slice_id, backend), embeddings)


def load_unaligned_vectors(slice_id: str, backend: str) -> dict[str, np.ndarray]:
    return _load_vectors(unaligned_vectors_path(slice_id, backend))


# Embeddings

def generate_embeddings_per_slice(
    slice_range: tuple[int, int],
    backend: str,
    force: bool = False
) -> dict[str, np.ndarray]:
    """
    Generate token embeddings for a given slice.

    Parameters
    ----------
    slice_range : tuple[int, int]
        Start and end years of the slice.
    backend : str
        Which embedding model to use. Currently supports:
        - "fasttext" : slice-trained fastText embeddings
        - "macberth" : placeholder for contextual BERT-style embeddings

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from token -> vector (float32)
    """
    if backend.lower() == "fasttext":
        return _generate_fasttext_embeddings(slice_range, force)
    elif backend.lower() == "macberth":
        return _generate_macberth_embeddings(slice_range, force)
    else:
        raise ValueError(f"Unknown embedding backend: {backend}")


# fastText backend
def _generate_fasttext_embeddings(slice_range: tuple[int, int], force:bool = False) -> dict[str, np.ndarray]:
    model_file = slice_model_path(slice_range, 'fasttext')
    slice_file = SLICES_DIR / f"{slice_range[0]}-{slice_range[1]}.txt"

    if not slice_file.exists():
        raise FileNotFoundError(f"Training corpus missing for slice {slice_range}: {slice_file}")

    if force or not model_file.exists():
        logger.info(f"Training fastText model for slice {slice_range} → {model_file}")
        model = fasttext.train_unsupervised(input=str(slice_file), **FASTTEXT_PARAMS)
        model.save_model(str(model_file))
    else:
        model = fasttext.load_model(str(model_file))

    return {str(tok): model.get_word_vector(tok).astype(np.float32) for tok in model.get_words()}


def has_fine_tuned_weights(ft_dir: Path) -> bool:
    return all((ft_dir / f).exists() for f in ["pytorch_model.bin", "config.json"])


# MacBERTh backend
def _forward_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: list[str],
    device: str
):
    # Add return_offsets_mapping=True
    batch_encoding = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )

    inputs = {k: v.to(device) for k, v in batch_encoding.items() if k != "offset_mapping"}

    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1].cpu().numpy()

    return hidden_states, batch_encoding


def _accumulate_tokens(
    tokenizer: PreTrainedTokenizerBase,
    batch_encoding,  # tokenizer(...) output with offsets
    hidden_states: np.ndarray,
    embeddings_accum: DefaultDict[str, List[np.ndarray]],
    sentences: List[str]
) -> None:
    """
    Accumulate embeddings per *true text span* using offset mappings.
    """

    input_ids = batch_encoding["input_ids"]
    offsets = batch_encoding["offset_mapping"]

    for b_idx, sent in enumerate(sentences):
        token_ids = input_ids[b_idx].tolist()
        token_offsets = offsets[b_idx].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        current_word = ""
        current_vecs: List[np.ndarray] = []
        last_end = None

        for idx, (_tok, (start, end)) in enumerate(zip(tokens, token_offsets, strict=True)):
            # Skip special tokens ([CLS], [SEP], etc.)
            if start == end:
                continue

            piece = sent[start:end]

            # Detect word boundary
            if last_end is not None and start != last_end:
                # finalize previous word
                if current_word and current_vecs:
                    vec = np.mean(np.stack(current_vecs, axis=0), axis=0)
                    embeddings_accum[current_word].append(vec)

                # reset
                current_word = ""
                current_vecs = []

            current_word += piece
            current_vecs.append(hidden_states[b_idx, idx])
            last_end = end

        # flush last word
        if current_word and current_vecs:
            vec = np.mean(np.stack(current_vecs, axis=0), axis=0)
            embeddings_accum[current_word].append(vec)


def _generate_macberth_embeddings(
    slice_range: Tuple[int, int],
    force: bool = False,
    batch_size: int = 128
) -> Dict[str, np.ndarray]:

    # Accumulate embeddings per token
    embeddings_accum: DefaultDict[str, List[np.ndarray]] = defaultdict(list)

    # Load model + tokenizer
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel

    tokenizer, model = get_macberth_model()  # model inferred as PreTrainedModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)  # don't reassign
    model.eval()

    # stream sentences
    conn = get_connection()
    sentence_stream = stream_slice_sentences(conn, slice_range)
    sentence_count = 0
    batch: List[str] = []

    with torch.no_grad():
        for sent in sentence_stream:
            batch.append(sent)
            if len(batch) < batch_size:
                continue

            hidden_states, batch_encoding = _forward_batch(model, tokenizer, batch, device)
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, batch)
            sentence_count += len(batch)
            if sentence_count % 500 == 0:
                logger.info("Processed %d sentences", sentence_count)
            batch.clear()

        # Process left overs
        if batch:
            hidden_states, batch_encoding = _forward_batch(model, tokenizer, batch, device)
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, batch)
            sentence_count += len(batch)

    conn.close()

    logger.info("Total sentences processed: %d", sentence_count)
    logger.info("Averaging embeddings for %d tokens", len(embeddings_accum))

    # Average embeddings per token
    final_embeddings: Dict[str, np.ndarray] = {
        tok: np.mean(np.stack(vlist, axis=0), axis=0).astype(np.float32)
        for tok, vlist in embeddings_accum.items()
    }

    logger.info("Generated embeddings for slice %d-%d", slice_range[0], slice_range[1])
    return final_embeddings


# Alignment

# Probably easier to outsource this
def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common:
        logger.warning("No anchors found; skipping alignment for this slice")
        return np.eye(len(next(iter(source_vectors.values()))), dtype=np.float32), source_vectors
    X = np.stack([source_vectors[w] for w in common])
    Y = np.stack([target_vectors[w] for w in common])
    R, _ = orthogonal_procrustes(X, Y)
    aligned = {w: vec @ R for w, vec in source_vectors.items()}
    # Normalize aligned vectors
    aligned = {k: v / np.linalg.norm(v) for k, v in aligned.items()}
    return R, aligned


# Public FAISS

def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    # cafeful of mypy
    add_fn = cast(Callable[[np.ndarray], None], index.add)
    add_fn(vectors)


def build_index_for_slice(
    slice_range: tuple[int,int],
    backend="macberth",
    use_aligned: bool = False,
    force: bool = False
) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned}, force={force})")

    # Determine paths
    index_path = faiss_slice_path(slice_range, use_aligned, backend)
    vocab_path = vocab_slice_path(slice_range, use_aligned, backend)

    # Load or regenerate embeddings
    if use_aligned:
        if force or not aligned_vectors_path(slice_id, backend).exists():
            embeddings = generate_embeddings_per_slice(slice_range, backend, force)
            save_aligned_vectors(slice_id, embeddings, backend)
        else:
            embeddings = load_aligned_vectors(slice_id, backend)
    else:
        if force or not unaligned_vectors_path(slice_id, backend).exists():
            embeddings = generate_embeddings_per_slice(slice_range, backend, force)
            save_unaligned_vectors(slice_id, embeddings, backend)
        else:
            embeddings = load_unaligned_vectors(slice_id, backend)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings for slice {slice_id}, skipping")
        return

    logger.info(f"Buiding FAISS index for slice {slice_id}")
    vectors = np.stack([embeddings[w] for w in words])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)


    vectors = np.stack([embeddings[w] for w in words])
    print("vectors type:", vectors.dtype, "shape:", vectors.shape)


    add_to_faiss_index(index, vectors)

    logger.info(f"Saving FAISS index and vocab for slice {slice_id}")
    faiss.write_index(index, str(index_path))
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logger.info(f"Saved FAISS index and vocab for slice {slice_id}")


def search_faiss(index: faiss.Index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    search_fn = cast(Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], index.search)
    return search_fn(query, k)


# Orchestration

def build_all_slices(backend: str, use_aligned: bool = False, force: bool = False) -> None:
    logger.info("Building slices (aligned=%s)", use_aligned)

    # produce amd save unaligned
    unaligned_cache: dict[str, dict[str, np.ndarray]] = {}

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        span = (start, end)

        try:
            vectors = load_unaligned_vectors(slice_id, backend)
            logger.info("Loaded unaligned vectors for %s", slice_id)
        except FileNotFoundError:
            vectors = generate_embeddings_per_slice(span, backend, force)
            save_unaligned_vectors(slice_id, vectors, backend)
            logger.info("Generated unaligned vectors for %s", slice_id)

        unaligned_cache[slice_id] = vectors
        build_index_for_slice(span, backend=backend, use_aligned=False, force=force)

    if not use_aligned:
        logger.info("Unaligned build complete.")
        return

    # align slices to center slice
    mid_index = len(SLICES) // 2
    reference_span = SLICES[mid_index]
    reference_id = f"{reference_span[0]}-{reference_span[1]}"
    ref_vectors = unaligned_cache[reference_id]

    anchors_dict = get_anchors()
    ref_anchors = anchors_dict[reference_id]["anchors"]

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        span = (start, end)

        if slice_id == reference_id:
            aligned_vectors = ref_vectors
        else:
            raw_vectors = unaligned_cache[slice_id]
            _, aligned_vectors = orthogonal_procrustes_align(
                raw_vectors,
                ref_vectors,
                ref_anchors,
            )

        save_aligned_vectors(slice_id, aligned_vectors, backend)
        build_index_for_slice(span, backend=backend, use_aligned=True, force=force)
        logger.info("Aligned and indexed %s", slice_id)

    logger.info("Aligned build complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS indexes per slice")
    parser.add_argument("--aligned", action="store_true", help="Use aligned slice embeddings")
    parser.add_argument("--force", action="store_true", help="Force regeneration of embeddings and indexes")

    args = parser.parse_args()
    env_aligned = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS", "").lower()
    use_aligned = args.aligned or env_aligned in {"1", "true", "yes"}

    env_force = os.environ.get("FORCE", "").lower()
    use_force = args.force or env_force in {"1", "true", "yes"}

    logger.info(f"Starting slice pipeline (forced={use_force}, aligned={use_aligned})")

    # build_all_slices(backend='macberth', use_aligned=use_aligned, force=use_force)
    build_all_slices(backend='macberth', use_aligned=True, force=True)


if __name__ == "__main__":
    main()
