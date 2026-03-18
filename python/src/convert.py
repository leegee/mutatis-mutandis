#!/usr/bin/env python
"""
convert_and_reindex.py

Convert old per-token-compressed `.npz` embeddings into contiguous
tokens+vectors format (2D float32), skipping non-numeric tokens,
then rebuild FAISS indexes for all slices.

Does NOT load or use MacBERTh.
"""

from pathlib import Path
import numpy as np
from slice_embedding_pipeline import SLICES, build_index_for_slice, MACBERTH_ALIGNED_VECTORS_DIR

OLD_DIR = MACBERTH_ALIGNED_VECTORS_DIR
NEW_DIR = MACBERTH_ALIGNED_VECTORS_DIR  # overwrite

def convert_old_npz(old_path: Path, new_path: Path):
    data = np.load(old_path, allow_pickle=True)
    tokens_list = []
    vectors_list = []

    print(f"Converting {old_path} ...")
    for k in data.files:
        try:
            vec = np.asarray(data[k], dtype=np.float32)
        except ValueError:
            print(f"Skipping non-numeric token: {k}")
            continue

        # Flatten extra dimensions, keeping last dim as embedding
        if vec.ndim > 1:
            vec = vec.reshape(-1, vec.shape[-1]).mean(axis=0)

        if vec.ndim != 1:
            print(f"Skipping token {k}, unexpected shape after flattening: {vec.shape}")
            continue

        tok = k.removeprefix("tok_")
        tokens_list.append(tok)
        vectors_list.append(vec)

    if not vectors_list:
        print(f"No valid numeric vectors found in {old_path}, skipping save")
        return

    vectors = np.stack(vectors_list, axis=0)  # shape (num_tokens, dim)
    np.savez_compressed(new_path, tokens=tokens_list, vectors=vectors)
    print(f"Saved new format to {new_path}")

def main():
    for start, end in SLICES:
        slice_id = f"{start}-{end}"
        old_file = OLD_DIR / f"{slice_id}.npz"
        new_file = NEW_DIR / f"{slice_id}.npz"

        if not old_file.exists():
            print(f"Skipping {slice_id}, file not found: {old_file}")
            continue

        convert_old_npz(old_file, new_file)

        # Rebuild FAISS index for this slice (unaligned)
        print(f"Rebuilding FAISS index for {slice_id}")
        # force=False ensures it loads your converted vectors
        build_index_for_slice((start, end), backend="macberth", use_aligned=False, force=False)

if __name__ == "__main__":
    main()
