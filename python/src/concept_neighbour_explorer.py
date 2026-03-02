#!/usr/bin/env python
"""
concept_neighbour_explorer.py

Exploratory concept neighbour & KWIC audit using vectors from slice_embedding_pipeline.

Defaults to aligned vectors (USE_ALIGNED_VECTORS=True).
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Tuple
import numpy as np

from lib.eebo_config import SLICES, CONCEPT_SETS, OUT_DIR
from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index
from slice_embedding_pipeline import load_aligned_vectors, load_unaligned_vectors, faiss_slice_path, vocab_slice_path

# --- Config ---
TARGET = 'LAW'  # Only audit this concept
USE_ALIGNED_VECTORS = os.environ.get("USE_ALIGNED_FASTTEXT_VECTORS", "1") == "1"

TOP_K = 100
SIM_THRESHOLD = 0.7
CONTEXT_WINDOW = 8
KWIC_MAX_LEFT = 40
KWIC_MAX_RIGHT = 40

json_path = OUT_DIR / f"concept_neighbour_audit_{TARGET.lower()}.json"
html_path = OUT_DIR / f"concept_kwic_audit_{TARGET.lower()}.html"

# --- Helper functions ---

def fetch_kwic_for_doc(conn, token: str, doc_id: str, limit: int = 3) -> List[Tuple[str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT token_idx FROM pamphlet_tokens WHERE doc_id = %s AND token = %s LIMIT %s",
            (doc_id, token, limit)
        )
        positions = [r[0] for r in cur.fetchall()]

    rows: List[Tuple[str, str, str]] = []
    if not positions:
        return rows

    min_idx = min(positions) - CONTEXT_WINDOW
    max_idx = max(positions) + CONTEXT_WINDOW

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_idx, token
            FROM pamphlet_tokens
            WHERE doc_id = %s AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx
            """,
            (doc_id, min_idx, max_idx),
        )
        context_rows = cur.fetchall()

    for idx in positions:
        left = " ".join(tok for i, tok in context_rows if i < idx)
        kw = next(tok for i, tok in context_rows if i == idx)
        right = " ".join(tok for i, tok in context_rows if i > idx)
        rows.append((left, kw, right))

    return rows

def make_html_row(left, kw, right, sim, freq, title, url):
    return f"""
    <tr>
        <td class="left">{left[:KWIC_MAX_LEFT]}</td>
        <td class="kw">{kw}</td>
        <td class="right">{right[:KWIC_MAX_RIGHT]}</td>
        <td>{sim:.3f}</td>
        <td>{freq}</td>
        <td><a href="{url}" target="_blank">{title}</a></td>
    </tr>
    """

def build_html(audit_data: Dict[str, Any]) -> str:
    parts = ["""
    <html><head>
    <style>
    body { font-family: monospace; background: black; color: #fffd }
    a { color: cyan }
    table { border-collapse: collapse; width: 100%; margin-bottom: 40px; }
    td, th { border: 1px solid #444; padding: 4px; }
    .kw { font-weight: bold; text-align: center; background: #ffe; color: black }
    .left { text-align: right; color: #aaa; }
    .right { text-align: left; color: #aaa; }
    </style></head><body>
    <h1>Concept Neighbour KWIC Audit (Canonicalised)</h1>
    """]

    for slice_key, concepts in audit_data.items():
        parts.append(f"<h2>Slice {slice_key}</h2>")
        for concept, block in concepts.items():
            seed = block["probe"]
            parts.append(f"<h3>Concept: {concept} (probe: {seed})</h3>")
            parts.append("<table><tr><th>Left</th><th>Keyword</th><th>Right</th><th>Sim</th><th>Freq</th><th>Doc</th></tr>")

            for n in block["neighbours"]:
                for doc in n["documents"]:
                    for left, kw, right in doc["kwic"]:
                        url = f"{getattr(OUT_DIR, 'TEXT_BASE_URL', '')}{doc['doc_id']}"
                        parts.append(make_html_row(
                            left, kw, right,
                            n["similarity"],
                            n["frequency"],
                            doc["title"],
                            url
                        ))
            parts.append("</table>")

    parts.append("</body></html>")
    return "\n".join(parts)


# --- Main loop ---
def main():
    logger.info("Starting concept neighbour explorer (aligned=%s)", USE_ALIGNED_VECTORS)
    audit: Dict[str, Any] = {}

    with get_connection() as conn:
        for slice_range in SLICES:
            slice_start, slice_end = slice_range
            slice_key = f"{slice_start}_{slice_end}"

            # Load FAISS index & vocab
            index, vocab = load_slice_index(slice_range, use_aligned=USE_ALIGNED_VECTORS)

            # Load vectors from pipeline file
            slice_id = f"{slice_start}-{slice_end}"
            try:
                if USE_ALIGNED_VECTORS:
                    vectors = load_aligned_vectors(slice_id)
                else:
                    vectors = load_unaligned_vectors(slice_id)
            except FileNotFoundError:
                logger.warning(f"No vectors for slice {slice_id}, skipping")
                continue

            audit[slice_key] = {}
            for concept, meta in CONCEPT_SETS.items():
                if concept != TARGET.upper():
                    continue

                seed = concept.lower()
                vec = vectors.get(seed)
                if vec is None:
                    logger.warning(f"No vector for probe '{seed}' in slice {slice_key}")
                    continue

                vec = vec / np.linalg.norm(vec)

                # --- FAISS search ---
                D, Idx = index.search(vec.reshape(1, -1), TOP_K)
                top_neighbors = [
                    (vocab[idx], float(sim))
                    for sim, idx in zip(D[0], Idx[0], strict=True)
                    if idx != -1
                ]

                # Exclude known forms
                known_forms = meta.get("forms", set())
                false_positives = meta.get("false_positives", set())
                top_neighbors = [(tok, sim) for tok, sim in top_neighbors if tok not in known_forms and tok not in false_positives]

                if not top_neighbors:
                    audit[slice_key][concept] = {"probe": seed, "neighbours": []}
                    continue

                tokens = [t for t, _ in top_neighbors]
                # --- Frequency lookup ---
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT token, COUNT(*) FROM pamphlet_tokens WHERE token = ANY(%s) AND slice_start = %s AND slice_end = %s GROUP BY token",
                        (tokens, slice_start, slice_end)
                    )
                    freq_map = {r[0]: r[1] for r in cur.fetchall()}

                # --- Document lookup ---
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT DISTINCT token, doc_id, title FROM pamphlet_tokens WHERE token = ANY(%s) AND slice_start = %s AND slice_end = %s",
                        (tokens, slice_start, slice_end)
                    )
                    docs_map: Dict[str, List[Dict[str, str]]] = {}
                    for token_, doc_id, title in cur.fetchall():
                        docs_map.setdefault(token_, []).append({"doc_id": doc_id, "title": title})

                # --- Fetch KWIC ---
                neighbours_list: List[Dict[str, Any]] = []
                for token, sim in top_neighbors:
                    token_freq = freq_map.get(token, 0)
                    docs: List[Dict[str, Any]] = docs_map.get(token, [])
                    for d in docs:
                        d["kwic"] = fetch_kwic_for_doc(conn, token, d["doc_id"])
                    neighbours_list.append({
                        "token": token,
                        "similarity": sim,
                        "frequency": token_freq,
                        "documents": docs
                    })

                audit[slice_key][concept] = {"probe": seed, "neighbours": neighbours_list}

    # --- Write outputs ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote {json_path}")

    html = build_html(audit)
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {html_path}")
    logger.info("Explorer complete.")


if __name__ == "__main__":
    main()
