#!/usr/bin/env python3

import time
from pathlib import Path
import numpy as np
from typing import Dict

from mb_embedding_pipeline import (
    load_vectors,
    faiss_slice_path,
    vocab_slice_path,
    id_map,
    FaissIndex,
)

SLICE = (1625, 1629)
SLICE_ID = f"{SLICE[0]}-{SLICE[1]}"


def wait_for_file(path: Path, label: str, poll: int = 5) -> None:
    print(f"[WAIT] Waiting for {label}: {path}")
    while not path.exists():
        time.sleep(poll)
    print(f"[OK] Found {label}")


def load_vocab(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def build_mean_vectors(embeddings_occ: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Compute mean vector per token for FAISS queries."""
    result: dict[str, np.ndarray] = {}
    for token, vecs in embeddings_occ.items():
        result[token] = np.mean(np.stack(vecs), axis=0).astype(np.float32)
    return result



def query(index: FaissIndex, embeddings: dict[str, np.ndarray], word_to_id: dict[str, int], id_to_word: dict[int,str], word: str, k: int = 10):
    if word not in embeddings:
        print(f"[MISS] '{word}' not in embeddings")
        return

    print(f"\nQuery: {word}")
    query_vec = normalize(embeddings[word]).reshape(1, -1)
    distances, indices = index.search(query_vec, k)

    for rank, numeric_id in enumerate(indices[0]):
        if numeric_id == -1:
            continue
        token = id_to_word.get(numeric_id, f"<unknown-{numeric_id}>")
        doc_id_val = id_map.index_to_doc_id.get(numeric_id, "<unknown-doc>")
        print(f"{rank}: {token} (doc {doc_id_val})  score={distances[0][rank]:.4f}")


def main() -> None:
    print(f"[INFO] Testing slice {SLICE_ID}")

    index_path = faiss_slice_path(SLICE)
    vocab_path = vocab_slice_path(SLICE)

    wait_for_file(index_path, "FAISS index")
    wait_for_file(vocab_path, "vocab")

    print("[INFO] Loading embeddings...")
    embeddings_occ = load_vectors(SLICE_ID)

    print("[INFO] Computing mean vectors for FAISS queries...")
    embeddings_mean = build_mean_vectors(embeddings_occ)

    print("[INFO] Loading FAISS index...")
    import faiss
    index = FaissIndex(len(next(iter(embeddings_mean.values()))))
    index._index = faiss.read_index(str(index_path))

    print("[INFO] Loading vocab...")
    words = load_vocab(vocab_path)

    # Stats
    vecs = np.stack(list(embeddings_mean.values()))
    norms = np.linalg.norm(vecs, axis=1)
    print(f"Vocab size: {len(embeddings_mean)}")
    print(f"Vector dim: {vecs.shape[1]}")
    print(f"Norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")

    # Queries
    test_words = ["god", "king", "church", "man", "love", "government", "liberty"]
    for w in test_words:
        query(index, embeddings_mean, words, w)

    print("\n[DONE] Test complete.")


if __name__ == "__main__":
    main()
