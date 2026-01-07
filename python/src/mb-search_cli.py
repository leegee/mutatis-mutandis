#!/usr/bin/env python3
"""
Command-line semantic search for the MacBERTh FAISS + SQLite index.

Usage:
    python search_cli.py "query here"
    python search_cli.py         # interactive mode
"""

import sys
import logging
from pathlib import Path

import numpy as np

from macberth_pipe.macberth_model import MacBERThModel
from macberth_pipe.semantic import SemanticIndex
from macberth_pipe.embedding import embed_chunks_batched
from macberth_pipe.types import Embeddings

# Paths (adapt as needed)
FAISS_STORE = Path("../../../faiss-cache/faiss-index")
SQLITE_DB   = Path("../../eebo-data/eebo-tcp_metadata.sqlite").resolve()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_cli")


def embed_query(model: MacBERThModel, text: str) -> np.ndarray:
    """Embed a single query text to a 1×D vector."""
    chunks = model.split_into_chunks(text, chunk_size=512)
    vecs = embed_chunks_batched(model, chunks, batch_size=4)
    if len(vecs) == 1:
        return vecs[0].reshape(1, -1)
    else:
        return np.mean(np.vstack(vecs), axis=0, keepdims=True)


def print_result(res: dict, colour: bool = True):
    """Pretty-print a search result."""
    rank = res["rank"]
    score = res["score"]
    doc_id = res.get("doc_id")
    chunk_idx = res.get("chunk_idx")

    title = res.get("title", "")
    author = res.get("author", "")
    year = res.get("year", "")
    text = res.get("text", "").replace("\n", " ").strip()
    text_preview = text[:200] + ("..." if len(text) > 200 else "")

    if colour:
        BOLD = "\033[1m"
        DIM  = "\033[2m"
        CYAN = "\033[96m"
        END  = "\033[0m"
    else:
        BOLD = DIM = CYAN = END = ""

    print(
        f"{BOLD}[{rank+1}] score={score:.4f}{END} "
        f"{CYAN}{title} ({author}, {year}){END}"
    )
    print(f"    doc_id = {doc_id}, chunk = {chunk_idx}")
    print(f"    {DIM}{text_preview}{END}")
    print()


def run_query(query: str):
    logger.info("Loading model…")
    model = MacBERThModel(device="cpu")

    logger.info("Embedding query…")
    qvec = embed_query(model, query)

    logger.info("Loading FAISS index…")
    idx = SemanticIndex(
        emb=Embeddings(ids=[], vectors=np.empty((0, 0)), metas=[]),
        store_dir=FAISS_STORE,
        sqlite_db=SQLITE_DB
    )

    logger.info("Searching…")
    results = idx.search(qvec, top_k=10)

    if not results:
        print("No results.")
        return

    for r in results:
        print_result(r)


def interactive():
    print("MacBERTh Semantic Search (interactive mode)")
    print("Type a query or 'quit' to exit.")
    print("-----------------------------------------")

    model = MacBERThModel(device="cpu")
    idx = SemanticIndex(
        emb=Embeddings(ids=[], vectors=np.empty((0, 0)), metas=[]),
        store_dir=FAISS_STORE,
        sqlite_db=SQLITE_DB,
    )

    while True:
        q = input("\nQuery > ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        print("Embedding...")
        qvec = embed_query(model, q)

        print("Searching...")
        results = idx.search(qvec, top_k=10)

        if not results:
            print("No results.")
            continue

        for r in results:
            print_result(r)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(query)
    else:
        interactive()


if __name__ == "__main__":
    main()
