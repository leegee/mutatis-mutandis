#!/usr/bin/env python
"""
audit_concept_neighbours.py

For each concept and each temporal slice:
- Find nearest semantic neighbours (FAISS)
- Count token frequency in that slice
- Retrieve documents containing the token
- Output JSON for corpus-cleaning and CONCEPT_SET refinement
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import numpy as np
import faiss

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from build_faiss_slice_indexes import faiss_slice_path, vocab_slice_path

TOP_K = 25  # neighbours per seed term


# TODO Move to the FAISS file
def load_slice_index(slice_range: Tuple[int, int]) -> tuple[Any, List[str]]:
    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)

    index = faiss.read_index(str(index_path))

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [w.strip() for w in f]

    return index, vocab


def get_vector(conn, token: str, start: int, end: int) -> np.ndarray | None:
    """Fetch canonical embedding from token_vectors table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT vector
            FROM token_vectors
            WHERE token = %s
              AND slice_start = %s
              AND slice_end   = %s
            LIMIT 1;
            """,
            (token, start, end),
        )
        row = cur.fetchone()

    if row is None:
        return None

    vec = np.array(row[0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else vec



# Corpus evidence
def token_frequency(conn, token: str, start: int, end: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM pamphlet_tokens
            WHERE token = %s
              AND slice_start = %s
              AND slice_end   = %s;
            """,
            (token, start, end),
        )
        return cur.fetchone()[0]


def token_documents(conn, token: str, start: int, end: int) -> List[Dict[str, Any]]:
    base_url = getattr(config, "TEXT_BASE_URL", "")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT d.doc_id, d.title
            FROM pamphlet_tokens t
            JOIN documents d ON d.doc_id = t.doc_id
            WHERE t.token = %s
              AND t.slice_start = %s
              AND t.slice_end   = %s
            LIMIT 20;
            """,
            (token, start, end),
        )
        rows = cur.fetchall()

    return [
        {
            "doc_id": doc_id,
            "title": title,
            "url": f"{base_url}{doc_id}" if base_url else None,
        }
        for doc_id, title in rows
    ]


# Main audit logic
def audit_slice(conn, slice_range: Tuple[int, int]) -> Dict[str, Any]:
    start, end = slice_range
    logger.info(f"Auditing slice {start}-{end}")

    index, vocab = load_slice_index(slice_range)
    vocab_arr = vocab

    slice_results: Dict[str, Any] = {}

    for concept_name, cfg in config.CONCEPT_SETS.items():
        slice_results[concept_name] = {}
        seeds = cfg["forms"]

        for seed in seeds:
            vec = get_vector(conn, seed, start, end)
            if vec is None:
                continue

            D, Idx = index.search(vec.reshape(1, -1), TOP_K)

            neighbours = []
            for sim, idx in zip(D[0], Idx[0], strict=True):
                token = vocab_arr[idx]
                freq = token_frequency(conn, token, start, end)
                docs = token_documents(conn, token, start, end)

                neighbours.append(
                    {
                        "token": token,
                        "similarity": float(sim),
                        "frequency": freq,
                        "documents": docs,
                    }
                )

            slice_results[concept_name][seed] = neighbours

    return slice_results


def main() -> None:
    logger.info("Starting neighbour audit")

    output: Dict[str, Any] = {}

    with eebo_db.get_connection() as conn:
        for slice_range in config.SLICES:
            key = f"{slice_range[0]}_{slice_range[1]}"
            output[key] = audit_slice(conn, slice_range)

    out_path = Path(config.OUT_DIR) / "concept_neighbour_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Audit complete → {out_path}")


if __name__ == "__main__":
    main()
