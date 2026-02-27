#!/usr/bin/env python
"""
concept_neighbour_explorer.py

Exploratory concept neighbour & KWIC audit with canonicalisation.

Each concept key is used as the probe (lowercased).
Known forms of the concept are excluded from the neighbors.

Hypothesis:

Expectations prior to running:

* In early EEBO slices: law constrains liberty (obedience, order).

* In the constitutional crisis: law enables liberty (rights, protections).

* In later slices: law stabilises liberty as something to be enjoyed or secured.

Outputs:

* concept_neighbour_audit.json
* concept_kwic_audit.html
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple
import numpy as np

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index, get_vector
from align import load_aligned_vectors

TARGET = 'LAW' # A key in config's CONCEPT_SETS
USE_ALIGNED_VECTORS = True

json_path = config.OUT_DIR / f"concept_neighbour_audit_{TARGET.lower()}.json"
html_path = config.OUT_DIR / f"concept_kwic_audit_{TARGET.lower()}.html"

TOP_K = 100
SIM_THRESHOLD = 0.7
CONTEXT_WINDOW = 8
KWIC_MAX_LEFT = 40
KWIC_MAX_RIGHT = 40


def fetch_kwic_for_doc(
    conn,
    token: str,
    doc_id: str,
    limit: int = 3,
) -> List[Tuple[str, str, str]]:
    """
    Fetch KWIC contexts for a token in a document, restricted to the
    pamphlet corpus.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_idx
            FROM pamphlet_tokens
            WHERE doc_id = %s
              AND token = %s
            LIMIT %s;
            """,
            (doc_id, token, limit),
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
            WHERE doc_id = %s
              AND token_idx BETWEEN %s AND %s
            ORDER BY token_idx;
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

            parts.append("""
            <table>
            <tr>
                <th>Left</th><th>Keyword</th><th>Right</th>
                <th>Sim</th><th>Freq</th><th>Doc</th>
            </tr>
            """)

            for n in block["neighbours"]:
                for doc in n["documents"]:
                    for left, kw, right in doc["kwic"]:
                        url = f"{getattr(config, 'TEXT_BASE_URL', '')}{doc['doc_id']}"
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



def main():
    logger.info("Starting canonicalised concept neighbour explorer")

    audit: Dict[str, Any] = {}

    with eebo_db.get_connection() as conn:
        for slice_range in config.SLICES:
            slice_start, slice_end = slice_range
            slice_key = f"{slice_start}_{slice_end}"

            # Load FAISS index and vocab
            index, vocab = load_slice_index(slice_range)
            audit[slice_key] = {}

            # Preload aligned vectors if requested
            aligned_vectors: dict[str, np.ndarray] | None = None
            if USE_ALIGNED_VECTORS:
                try:
                    aligned_vectors = load_aligned_vectors(f"{slice_start}-{slice_end}")
                    logger.info(f"Loaded aligned vectors for slice {slice_key}")
                except FileNotFoundError:
                    logger.warning(f"No aligned vectors for slice {slice_key}, falling back to FAISS")
                    aligned_vectors = None

            for concept in config.CONCEPT_SETS.keys():
                if concept != TARGET.upper():
                    continue

                seed = concept.lower()

                # --- Get probe vector ---
                if USE_ALIGNED_VECTORS and aligned_vectors is not None:
                    vec = aligned_vectors.get(seed)
                    if vec is None:
                        logger.warning(f"No aligned vector for probe '{seed}' in slice {slice_key}")
                        continue
                else:
                    vec = get_vector(conn, seed, slice_start, slice_end)
                    if vec is None:
                        logger.warning(f"No vector for probe '{seed}' in slice {slice_key}")
                        continue

                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                logger.info(f"Processing slice {slice_key}, concept {concept}")

                # --- FAISS search ---
                D, Idx = index.search(vec.reshape(1, -1), TOP_K)
                top_neighbors = [
                    (vocab[idx], float(sim))
                    for sim, idx in zip(D[0], Idx[0], strict=True)
                    if idx != -1
                ]

                # Exclude known forms / false positives
                known_forms = config.CONCEPT_SETS[concept].get("forms", set())
                false_positives = config.CONCEPT_SETS[concept].get("false_positives", set())
                top_neighbors = [
                    (token, sim)
                    for token, sim in top_neighbors
                    if token not in known_forms and token not in false_positives
                ]

                if not top_neighbors:
                    audit[slice_key][concept] = {"probe": seed, "neighbours": []}
                    continue

                # --- Batch frequency lookup ---
                tokens = [t for t, _ in top_neighbors]
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT token, COUNT(*)
                        FROM pamphlet_tokens
                        WHERE token = ANY(%s)
                        AND slice_start = %s
                        AND slice_end = %s
                        GROUP BY token
                        """,
                        (tokens, slice_start, slice_end),
                    )
                    freq_map = {r[0]: r[1] for r in cur.fetchall()}

                # --- Batch document lookup ---
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT DISTINCT token, doc_id, title
                        FROM pamphlet_tokens
                        WHERE token = ANY(%s)
                        AND slice_start = %s
                        AND slice_end = %s
                        """,
                        (tokens, slice_start, slice_end),
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

                audit[slice_key][concept] = {
                    "probe": seed,
                    "neighbours": neighbours_list
                }

    # --- Write JSON ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote {json_path}")

    # --- Write HTML ---
    html = build_html(audit)
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {html_path}")

    logger.info("Explorer complete.")

if __name__ == "__main__":
    main()
