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
from typing import Callable, cast, DefaultDict
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
    FASTTEXT_PARAMS,
    FASTTEXT_SLICE_MODEL_DIR,
    SLICES, SLICES_DIR,
    FASTTEXT_UNALIGNED_VECTORS_DIR,
    FASTTEXT_ALIGNED_VECTORS_DIR,
    EEBO_MODEL_NAME,
    MACBERTH_FINE_TUNED_DIR
)
from lib.eebo_anchor_builder import get_anchors
from lib.eebo_sentences import stream_slice_sentences


# nltk.download('punkt')

# Cache
TOKENIZER: PreTrainedTokenizerBase | None = None
MODEL: PreTrainedModel | None = None

# Paths

def unaligned_vectors_path(slice_id: str) -> Path:
    return FASTTEXT_UNALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def aligned_vectors_path(slice_id: str) -> Path:
    return FASTTEXT_ALIGNED_VECTORS_DIR / f"{slice_id}.npz"


def slice_model_path(slice_range: tuple[int,int]) -> Path:
    start, end = slice_range
    return FASTTEXT_SLICE_MODEL_DIR / f"slice_{start}_{end}.bin"


def faiss_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = (FASTTEXT_ALIGNED_VECTORS_DIR if aligned else FASTTEXT_UNALIGNED_VECTORS_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)
    start, end = slice_range
    return base_dir / f"slice_{start}_{end}.faiss"


def vocab_slice_path(slice_range: tuple[int,int], aligned: bool) -> Path:
    base_dir = (FASTTEXT_ALIGNED_VECTORS_DIR if aligned else FASTTEXT_UNALIGNED_VECTORS_DIR)
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
    # Prefix keys to avoid NumPy argument collision
    safe_dict = {f"tok_{k}": v for k, v in embeddings.items()}
    np.savez_compressed(path, **safe_dict)


def _load_vectors(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Vectors missing: {path}")
    with np.load(path) as data:
        return {
            k.removeprefix("tok_"): data[k].astype(np.float32)
            for k in data
        }

# Public IO

def save_aligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    _save_vectors(aligned_vectors_path(slice_id), embeddings)


def load_aligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    return _load_vectors(aligned_vectors_path(slice_id))


def save_unaligned_vectors(slice_id: str, embeddings: dict[str, np.ndarray]) -> None:
    _save_vectors(unaligned_vectors_path(slice_id), embeddings)


def load_unaligned_vectors(slice_id: str) -> dict[str, np.ndarray]:
    return _load_vectors(unaligned_vectors_path(slice_id))


# Embeddings

# def generate_embeddings_per_slice(slice_range: tuple[int,int]) -> dict[str, np.ndarray]:
#     model_file = slice_model_path(slice_range)
#     if not model_file.exists():
#         slice_file = SLICES_DIR / f"{slice_range[0]}-{slice_range[1]}.txt"
#         if not slice_file.exists():
#             raise FileNotFoundError(f"Training corpus missing for slice {slice_range}: {slice_file}")
#         logger.info(f"Training fastText model for slice {slice_range} → {model_file}")
#         model = fasttext.train_unsupervised(input=str(slice_file), **FASTTEXT_PARAMS)
#         model.save_model(str(model_file))
#     else:
#         model = fasttext.load_model(str(model_file))
#     return {str(tok): model.get_word_vector(tok).astype(np.float32) for tok in model.get_words()}

def generate_embeddings_per_slice(
    slice_range: tuple[int, int],
    backend: str = "fasttext",
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
    model_file = slice_model_path(slice_range)
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
def _forward_batch(model, tokenizer, batch, device):
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1].cpu().numpy()
    input_ids = inputs["input_ids"].cpu()

    return hidden_states, input_ids


def _accumulate_tokens(tokenizer, input_ids, hidden_states, embeddings_accum):
    for b_idx in range(len(input_ids)):

        tokens = tokenizer.convert_ids_to_tokens(input_ids[b_idx])

        word_buffer: list[str] = []
        vec_buffer: list[np.ndarray] = []

        for t, vec in zip(tokens, hidden_states[b_idx], strict=True):
            if t in {"[CLS]", "[SEP]", "[PAD]"}:
                continue

            if t.startswith("##"):
                if word_buffer:
                    word_buffer[-1] += t[2:]
                    vec_buffer[-1] += vec
            else:
                word_buffer.append(t)
                vec_buffer.append(vec)

        for w, v in zip(word_buffer, vec_buffer, strict=True):
            embeddings_accum[w].append(v)


def _generate_macberth_embeddings(
    slice_range: tuple[int, int],
    force: bool = False,
    batch_size: int = 32
) -> dict[str, np.ndarray]:

    embeddings_accum: DefaultDict[str, list[np.ndarray]] = defaultdict(list)

    tokenizer, model = get_macberth_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cast(torch.nn.Module, model).to(device)

    conn = get_connection()
    logger.info( "Streaming sentences from DB for slice %d-%d", slice_range[0], slice_range[1] )
    sentence_stream = stream_slice_sentences(conn, slice_range)

    sentence_count = 0
    batch: list[str] = []

    with torch.no_grad():
        for sent in sentence_stream:
            batch.append(sent)

            if len(batch) < batch_size:
                continue

            hidden_states, input_ids = _forward_batch(
                model, tokenizer, batch, device
            )

            _accumulate_tokens( tokenizer, input_ids, hidden_states, embeddings_accum )

            sentence_count += len(batch)

            if sentence_count % 5000 == 0:
                logger.info("Processed %d sentences", sentence_count)

            batch.clear()

        if batch:
            hidden_states, input_ids = _forward_batch(
                model, tokenizer, batch, device
            )

            _accumulate_tokens(
                tokenizer,
                input_ids,
                hidden_states,
                embeddings_accum
            )

            sentence_count += len(batch)

    conn.close()

    logger.info("Total sentences processed: %d", sentence_count)
    logger.info(
        "Averaging embeddings for %d tokens",
        len(embeddings_accum)
    )

    final_embeddings = {
        tok: np.mean(np.stack(vlist, axis=0), axis=0).astype(np.float32)
        for tok, vlist in embeddings_accum.items()
    }

    logger.info( "Generated embeddings for slice %d-%d", slice_range[0], slice_range[1] )
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
    backend="fasttext",
    use_aligned: bool = False,
    force: bool = False
) -> None:
    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    logger.info(f"Processing slice {slice_id} (aligned={use_aligned}, force={force})")

    # Determine paths
    index_path = faiss_slice_path(slice_range, use_aligned)
    vocab_path = vocab_slice_path(slice_range, use_aligned)

    # Load or regenerate embeddings
    if use_aligned:
        if force or not aligned_vectors_path(slice_id).exists():
            embeddings = generate_embeddings_per_slice(slice_range, backend, force)
            save_aligned_vectors(slice_id, embeddings)
        else:
            embeddings = load_aligned_vectors(slice_id)
    else:
        if force or not unaligned_vectors_path(slice_id).exists():
            embeddings = generate_embeddings_per_slice(slice_range, backend, force)
            save_unaligned_vectors(slice_id, embeddings)
        else:
            embeddings = load_unaligned_vectors(slice_id)

    words = list(embeddings.keys())
    if not words:
        logger.warning(f"No embeddings for slice {slice_id}, skipping")
        return

    # Build FAISS index
    vectors = np.stack([embeddings[w] for w in words])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    add_to_faiss_index(index, vectors)

    # Save index and vocab (overwrite if force=True)
    faiss.write_index(index, str(index_path))
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(words))
    logger.info(f"Saved FAISS index and vocab for slice {slice_id}")


def search_index(index: faiss.Index, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    search_fn = cast(Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], index.search)
    return search_fn(query, k)


# Orchestration

def build_all_slices(use_aligned: bool = False, force: bool = False, backend = "fasttext") -> None:
    logger.info("Building slices (aligned=%s)", use_aligned)

    # Stage 1: produce & save unaligned
    unaligned_cache: dict[str, dict[str, np.ndarray]] = {}

    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        span = (start, end)

        try:
            vectors = load_unaligned_vectors(slice_id)
            logger.info("Loaded unaligned vectors for %s", slice_id)
        except FileNotFoundError:
            vectors = generate_embeddings_per_slice(span, backend, force)
            save_unaligned_vectors(slice_id, vectors)
            logger.info("Generated unaligned vectors for %s", slice_id)

        unaligned_cache[slice_id] = vectors
        build_index_for_slice(span, backend=backend, use_aligned=False, force=force)

    if not use_aligned:
        logger.info("Unaligned build complete.")
        return

    # Stage 2: align slices to center slice
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

        save_aligned_vectors(slice_id, aligned_vectors)
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

    logger.info(f"Starting slice pipeline (aligned={use_aligned})")

    build_all_slices(use_aligned=use_aligned, backend='macberth', force=use_force) # fasttext


if __name__ == "__main__":
    main()
