#!/usr/bin/env python3

import time
from pathlib import Path
import numpy as np
import faiss

from mb_embedding_pipeline import (
    load_vectors,
    faiss_slice_path,
    vocab_slice_path,
    id_map,
)

SLICE = (1625, 1629)
SLICE_ID = f"{SLICE[0]}-{SLICE[1]}"

def wait_for_file(path: Path, label: str, poll: int = 5):
    print(f"[WAIT] Waiting for {label}: {path}")
    while not path.exists():
        time.sleep(poll)
    print(f"[OK] Found {label}")


def load_vocab(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def build_word_index(words: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(words)}


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def query(
    index: faiss.Index,
    embeddings: dict[str, np.ndarray],
    words: list[str],
    word_to_idx: dict[str, int],
    word: str,
    k: int = 10
):
    if word not in embeddings:
        print(f"[MISS] '{word}' not in embeddings")
        return

    print(f"\nQuery: {word}")

    query_vec = normalize(embeddings[word]).astype("float32").reshape(1, -1)
    D, _I = index.search(query_vec, k)

    for rank, numeric_id in enumerate(_I[0]):
        doc_id = id_map.index_to_doc_id[numeric_id]   # retrieve original EEBO doc ID
        print(f"{rank}: {words[numeric_id]} (doc {doc_id})  score={D[0][rank]:.4f}")


def main():
    print(f"[INFO] Testing slice {SLICE_ID}")

    index_path = faiss_slice_path(SLICE)
    vocab_path = vocab_slice_path(SLICE)

    # Wait until ingestion produces outputs
    wait_for_file(index_path, "FAISS index")
    wait_for_file(vocab_path, "vocab")

    print("[INFO] Loading embeddings...")
    embeddings = load_vectors(SLICE_ID)

    print("[INFO] Loading FAISS index...")
    index = faiss.read_index(str(index_path))

    print("[INFO] Loading vocab...")
    words = load_vocab(vocab_path)
    word_to_idx = build_word_index(words)

    # Basic checks
    print("\n[CHECK] Embedding stats")
    vecs = np.stack(list(embeddings[w] for w in words))
    norms = np.linalg.norm(vecs, axis=1)

    print(f"Vocab size: {len(embeddings)}")
    print(f"Vector dim: {vecs.shape[1]}")
    print(f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # Sample tokens
    print("\n[CHECK] Sample tokens:")
    for w in words[:20]:
        print(" ", w)

    # Queries
    test_words = ["god", "king", "church", "man", "love","government", "liberty"]
    for w in test_words:
        query(index, embeddings, words, word_to_idx, w)

    print("\n[DONE] Test complete.")


if __name__ == "__main__":
    main()
