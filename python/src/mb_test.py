#!/usr/bin/env python3

import time
from pathlib import Path
import numpy as np
import faiss

from slice_embedding_pipeline import (
    load_unaligned_vectors,
    faiss_slice_path,
    vocab_slice_path,
)

SLICE = (1625, 1629)
SLICE_ID = f"{SLICE[0]}-{SLICE[1]}"
BACKEND = "macberth"
ALIGNED = False


def wait_for_file(path: Path, label: str, poll=5):
    print(f"[WAIT] Waiting for {label}: {path}")
    while not path.exists():
        time.sleep(poll)
    print(f"[OK] Found {label}")


def load_vocab(path: Path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def build_word_index(words):
    return {w: i for i, w in enumerate(words)}


def normalize(v):
    return v / np.linalg.norm(v)


def query(index, embeddings, words, word_to_idx, word, k=10):
    if word not in embeddings:
        print(f"[MISS] '{word}' not in embeddings")
        return

    vec = normalize(embeddings[word]).astype("float32").reshape(1, -1)
    D, _I = index.search(vec, k)

    print(f"\nQuery: {word}")
    for rank, idx in enumerate(_I[0]):
        print(f"{rank:2d}  {words[idx]:20s}  {D[0][rank]:.4f}")


def main():
    print(f"[INFO] Testing slice {SLICE_ID}")

    index_path = faiss_slice_path(SLICE, ALIGNED, BACKEND)
    vocab_path = vocab_slice_path(SLICE, ALIGNED, BACKEND)

    # Wait until ingestion produces outputs
    wait_for_file(index_path, "FAISS index")
    wait_for_file(vocab_path, "vocab")

    print("[INFO] Loading embeddings...")
    embeddings = load_unaligned_vectors(SLICE_ID, BACKEND)

    print("[INFO] Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("[INFO] Loading vocab...")
    words = load_vocab(vocab_path)
    word_to_idx = build_word_index(words)

    # --- Basic checks ---
    print("\n[CHECK] Embedding stats")
    vecs = np.stack(list(embeddings.values()))
    norms = np.linalg.norm(vecs, axis=1)

    print(f"Vocab size: {len(embeddings)}")
    print(f"Vector dim: {vecs.shape[1]}")
    print(f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # --- Sample tokens ---
    print("\n[CHECK] Sample tokens:")
    for w in list(embeddings.keys())[:20]:
        print(" ", w)

    # --- Queries ---
    test_words = ["god", "king", "church", "man", "love"]

    for w in test_words:
        query(index, embeddings, words, word_to_idx, w)

    print("\n[DONE] Test complete.")


if __name__ == "__main__":
    main()
