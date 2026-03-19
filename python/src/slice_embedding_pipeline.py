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
from typing import Callable, cast, DefaultDict, List, Dict, Tuple, Optional, Union
from collections import defaultdict

import fasttext
import faiss
import numpy as np
from scipy.linalg import orthogonal_procrustes
import torch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
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
from lib.eebo_id_map import EEBOIDMap


# Cache
TOKENIZER: Optional[PreTrainedTokenizerBase] = None
MODEL: Optional[PreTrainedModel] = None

id_map = EEBOIDMap()
id_map.load()


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
    slice_range: tuple[int,int],
    backend: str = "macberth",
    force: bool = False,
    reference_slice_id: str | None = None
) -> tuple[dict[str, np.ndarray], DefaultDict[str, list[str]]]:
    """
    Generate token embeddings for a given slice. Optionally align using anchors from a reference slice.

    Returns:
        embeddings: dict[token -> vector]
        doc_ids_accum: mapping token -> list of doc_ids
    """
    slice_id = f"{slice_range[0]}-{slice_range[1]}"

    if backend.lower() == "fasttext":
        embeddings = _generate_fasttext_embeddings(slice_range, force)
        doc_ids_accum: DefaultDict[str, list[str]] = defaultdict(list)
        # If reference_slice_id is provided, apply orthogonal alignment
        if reference_slice_id:
            anchors = get_anchors()
            _, embeddings = orthogonal_procrustes_align(embeddings, load_aligned_vectors(reference_slice_id, backend), anchors)
        return embeddings, doc_ids_accum

    elif backend.lower() == "macberth":
        embeddings, doc_ids_accum = _generate_macberth_embeddings(slice_range, force)

        if reference_slice_id:
            logger.info(f"Aligning slice {slice_id} to reference slice {reference_slice_id}")
            ref_embeddings = load_aligned_vectors(reference_slice_id, backend)
            anchors = get_anchors()
            _, embeddings = orthogonal_procrustes_align(embeddings, ref_embeddings, anchors)

        return embeddings, doc_ids_accum

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
) -> Tuple[np.ndarray, BatchEncoding]:
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
    doc_ids_accum: DefaultDict[str, List[str]],
    sentences: List[Tuple[str,str]]
) -> None:
    """
    Accumulate embeddings per *true text span* and track doc IDs for FAISS.
    Uses strict=True for zip to ensure token/offset alignment.
    """
    input_ids = batch_encoding["input_ids"]
    offsets = batch_encoding["offset_mapping"]

    for b_idx, (doc_id, sent) in enumerate(sentences):
        token_ids = input_ids[b_idx].tolist()
        token_offsets = offsets[b_idx].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        current_word = ""
        current_vecs: List[np.ndarray] = []
        last_end = None

        if len(tokens) != len(token_offsets):
            raise ValueError(f"Mismatch: {len(tokens)} tokens vs {len(token_offsets)} offsets")

        # Strict zip ensures no length mismatch silently passes
        for idx, (_tok, (start, end)) in enumerate(zip(tokens, token_offsets, strict=True)):
            if start == end:
                continue

            piece = sent[start:end]

            # New word boundary
            if last_end is not None and start != last_end:
                if current_word and current_vecs:
                    vec = np.mean(np.stack(current_vecs, axis=0), axis=0)
                    embeddings_accum[current_word].append(vec)
                    doc_ids_accum[current_word].append(doc_id)
                current_word = ""
                current_vecs = []

            current_word += piece
            current_vecs.append(hidden_states[b_idx, idx])
            last_end = end

        # Flush last word
        if current_word and current_vecs:
            vec = np.mean(np.stack(current_vecs, axis=0), axis=0)
            embeddings_accum[current_word].append(vec)
            doc_ids_accum[current_word].append(doc_id)



def _generate_macberth_embeddings(
    slice_range: Tuple[int, int],
    force: bool = False,
    batch_size: int = 128
) -> Tuple[Dict[str, np.ndarray], DefaultDict[str, List[str]]]:

    # Accumulate embeddings per token
    embeddings_accum: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
    doc_ids_accum: DefaultDict[str, List[str]] = defaultdict(list)

    # Load model + tokenizer
    tokenizer, model = get_macberth_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Stream sentences from DB
    conn = get_connection()
    sentence_stream = stream_slice_sentences(conn, slice_range)  # yields (doc_id, sentence)
    sentence_count = 0
    batch: List[Tuple[str, str]] = []

    with torch.no_grad():
        for sent_tuple in sentence_stream:  # sent_tuple = (doc_id, sentence)
            batch.append(sent_tuple)
            if len(batch) < batch_size:
                continue

            hidden_states, batch_encoding = _forward_batch(
                model, tokenizer, [s for _, s in batch], device
            )
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, doc_ids_accum, batch)
            sentence_count += len(batch)
            if sentence_count % 500 == 0:
                logger.info("Processed %d sentences", sentence_count)
            batch.clear()

        # Process leftovers
        if batch:
            hidden_states, batch_encoding = _forward_batch(
                model, tokenizer, [s for _, s in batch], device
            )
            _accumulate_tokens(tokenizer, batch_encoding, hidden_states, embeddings_accum, doc_ids_accum, batch)
            sentence_count += len(batch)

    conn.close()

    logger.info("Total sentences processed: %d", sentence_count)
    logger.info("Averaging embeddings for %d tokens", len(embeddings_accum))

    # Convert lists to single embeddings (mean)
    final_embeddings: Dict[str, np.ndarray] = {}
    for token, vecs in embeddings_accum.items():
        if isinstance(vecs, list) and vecs and isinstance(vecs[0], np.ndarray):
            final_embeddings[token] = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)

    logger.info("Generated embeddings for slice %d-%d", slice_range[0], slice_range[1])
    return final_embeddings, doc_ids_accum


# Alignment

def orthogonal_procrustes_align(source_vectors, target_vectors, anchor_words):
    common = [w for w in anchor_words if w in source_vectors and w in target_vectors]
    if not common:
        logger.warning("No anchors found; skipping alignment for this slice")
        return np.eye(len(next(iter(source_vectors.values()))), dtype=np.float32), source_vectors
    X = np.stack([source_vectors[w] for w in common])
    Y = np.stack([target_vectors[w] for w in common])
    R, _ = orthogonal_procrustes(X, Y)

    # Apply rotation
    aligned_vectors = {k: vec @ R for k, vec in source_vectors.items()}

    # Normalize
    aligned_vectors = {k: v / (np.linalg.norm(v) + 1e-10) for k, v in aligned_vectors.items()}

    return R, aligned_vectors


# Public FAISS

# def add_to_faiss_index(index: faiss.Index, vectors: np.ndarray) -> None:
#     vectors = np.ascontiguousarray(vectors, dtype=np.float32)
#     # cafeful of mypy
#     add_fn = cast(Callable[[np.ndarray], None], index.add)
#     add_fn(vectors)

def add_to_faiss_index(
    index: faiss.Index,
    vectors: np.ndarray,
    vector_ids: Optional[Union[np.ndarray, list[int]]] = None
) -> None:
    """
    Add vectors to a FAISS index, optionally with explicit IDs (e.g., doc_id).

    Parameters
    ----------
    index : faiss.Index
        FAISS index (should be wrapped in IndexIDMap if using IDs)
    vectors : np.ndarray
        2D array of shape (num_vectors, dim)
    vector_ids : np.ndarray or list[int], optional
        IDs for each vector (e.g., doc_id). Must match vectors.shape[0]
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


def build_index_for_slice(
    slice_range: tuple[int,int],
    backend: str = "macberth",
    use_aligned: bool = False,
    force: bool = False,
    reference_slice_id: str | None = None
) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned}, force={force})")

    # Paths
    index_path = faiss_slice_path(slice_range, use_aligned, backend)
    vocab_path = vocab_slice_path(slice_range, use_aligned, backend)
    vectors_path = aligned_vectors_path(slice_id, backend) if use_aligned else unaligned_vectors_path(slice_id, backend)

    # Load or generate embeddings + doc_ids
    if force or not vectors_path.exists():
        embeddings, doc_ids_accum = generate_embeddings_per_slice(
            slice_range,
            backend,
            force,
            reference_slice_id=reference_slice_id
        )
        if use_aligned:
            save_aligned_vectors(slice_id, embeddings, backend)
        else:
            save_unaligned_vectors(slice_id, embeddings, backend)
    else:
        if use_aligned:
            embeddings = load_aligned_vectors(slice_id, backend)
        else:
            embeddings = load_unaligned_vectors(slice_id, backend)
        doc_ids_accum = defaultdict(list)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings for slice {slice_id}, skipping index build")
        return

    # Flatten embeddings and assign numeric IDs
    all_vectors = []
    all_ids = []

    for word in words:
        vecs_list = embeddings[word] if isinstance(embeddings[word], list) else [embeddings[word]]
        doc_ids_list = doc_ids_accum[word] if word in doc_ids_accum else list(range(len(vecs_list)))
        for vec, doc_id in zip(vecs_list, doc_ids_list, strict=True):
            all_vectors.append(vec)
            all_ids.append(id_map.get_numeric_id(doc_id))

    vectors = np.stack(all_vectors).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)  # normalize
    vector_ids = np.array(all_ids, dtype=np.int64)

    # Build FAISS index
    dim = vectors.shape[1]
    base_index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(base_index)
    add_to_faiss_index(index, vectors, vector_ids)

    # Save FAISS index
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index at {index_path}")

    # Save vocab
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logger.info(f"Saved vocab at {vocab_path}")

    id_map.save()
    logger.info("Saved EEBO ID map")
    logger.info("FAISS build complete.")


def build_all_slices(
    backend: str = "macberth",
    use_aligned: bool = False,
    force: bool = False,
    reference_slice_id: str | None = None
) -> None:
    """
    Build FAISS indexes for all slices, optionally aligned to a reference slice.
    """
    for start, end in SLICES:
        build_index_for_slice(
            slice_range=(start, end),
            backend=backend,
            use_aligned=use_aligned,
            force=force,
            reference_slice_id=reference_slice_id
        )


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and FAISS indexes per slice")
    parser.add_argument("--aligned", action="store_true", help="Use aligned slice embeddings")
    parser.add_argument("--force", action="store_true", help="Force regeneration of embeddings and indexes")
    parser.add_argument("--reference-slice", type=str, default=None, help="Reference slice ID for anchor alignment (e.g., '1625-1629')")

    args = parser.parse_args()
    env_aligned = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS", "").lower()
    use_aligned = args.aligned or env_aligned in {"1", "true", "yes"}

    env_force = os.environ.get("FORCE", "").lower()
    use_force = args.force or env_force in {"1", "true", "yes"}

    env_backend = os.environ.get("BACKEND", "").lower()
    use_backend = args.backend or "macberth"

    start, end = SLICES[len(SLICES)//2]  # middle slice
    reference_slice_id = args.reference_slice or f"{start}-{end}"


    use_backend="macberth"
    use_force=True
    use_aligned=True

    logger.info(f"Starting slice pipeline (force={use_force}, aligned={use_aligned}, reference={reference_slice_id})")

    build_all_slices(
        backend='macberth',
        use_aligned=use_aligned,
        force=use_force,
        reference_slice_id=reference_slice_id
    )

if __name__ == "__main__":
    main()
