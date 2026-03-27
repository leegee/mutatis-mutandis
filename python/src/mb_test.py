#!/usr/bin/env python3
"""
Minimal sanity test for occurrence-level and token-level FAISS indexes.

- Uses wrapper classes only (no direct faiss usage)
- Runs on first slice only
- Confirms both indexes load and return results
"""

from pathlib import Path

from lib.FaissIndex import FaissIndex
from lib.TokenFaissIndex import TokenFaissIndex
from mb_embedding_pipeline import (
    faiss_slice_path,
    token_list_path,
    load_model_for_slice,
    embed_word_with_model
)

SLICE = (1642, 1642)
SLICE_ID = f"{SLICE[0]}-{SLICE[1]}"


def load_token_list(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main() -> None:
    print(f"[INFO] Testing slice {SLICE_ID}")

    # paths
    occ_index_path = faiss_slice_path(SLICE)
    token_index_path = occ_index_path.with_name(
        f"slice_{SLICE[0]}_{SLICE[1]}.token.faiss"
    )
    token_list_file = token_list_path(SLICE[0], SLICE[1])

    # load indexes via wrappers
    print("[INFO] Loading occurrence-level index...")
    occ_index = FaissIndex.load(str(occ_index_path))

    print("[INFO] Loading token-level index...")
    token_index = TokenFaissIndex.load(str(token_index_path))

    print("[INFO] Loading token list...")
    tokens_ordered = load_token_list(token_list_file)

    # invariant check
    if len(tokens_ordered) != token_index.ntotal:
        raise ValueError(
            f"Token list ({len(tokens_ordered)}) != index size ({token_index.ntotal})"
        )

    # load embeddings (for query vectors only)
    # print("[INFO] Loading embeddings...")
    # embeddings_occ = load_vectors(SLICE_ID)
    #
    # embeddings_mean = {
    #     tok: np.mean(np.stack(vecs), axis=0).astype(np.float32)
    #     for tok, vecs in embeddings_occ.items()
    # }
    #
    # print(f"[INFO] Vocab size: {len(embeddings_mean)}")

    # test queries
    test_words = ["god", "king", "church", "man", "sword", "ship", "bread", "horse"]

    model, tokenizer = load_model_for_slice(SLICE[0], SLICE[1])

    for word in test_words:
        print(f"\n=== {word} ===")

        # if word not in embeddings_mean:
        #     print("[MISS]")
        #     continue

        q = embed_word_with_model(word, model, tokenizer).reshape(1, -1)

        # occurrence-level
        d_occ, i_occ = occ_index.search(q, k=5)
        print("[occurrence]")
        for rank, idx in enumerate(i_occ[0]):
            if idx == -1:
                continue
            print(f"{rank}: id={idx} score={d_occ[0][rank]:.4f}")

        # token-level
        d_tok, i_tok = token_index.search(q, k=5)
        print("[token]")
        for rank, idx in enumerate(i_tok[0]):
            if idx == -1:
                continue
            token = tokens_ordered[idx]
            print(f"{rank}: {token} score={d_tok[0][rank]:.4f}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
