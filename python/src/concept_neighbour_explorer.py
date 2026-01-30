#!/usr/bin/env python
"""
concept_neighbour_explorer.py

Orthographic and scribal variants of the same lexeme (eg “liberty”),

Outputs:
- concept_neighbour_audit.json
- concept_kwic_audit.html
"""

from __future__ import annotations
import json
from typing import Any, Dict, List
import time
import numpy as np

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index, get_vector

TOP_K = 25
SIM_THRESHOLD = 0.75
CONTEXT_WINDOW = 8
LOG_INTERVAL = 120

# DB helpers
def fetch_token_frequency(conn, token: str, slice_start: int, slice_end: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM pamphlet_tokens
            WHERE token = %s
              AND slice_start = %s
              AND slice_end = %s;
            """,
            (token, slice_start, slice_end),
        )
        return cur.fetchone()[0]


def fetch_documents_for_token(conn, token: str, slice_start: int, slice_end: int, limit: int = 5):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT doc_id, title
            FROM pamphlet_tokens
            WHERE token = %s
              AND slice_start = %s
              AND slice_end = %s
            LIMIT %s;
            """,
            (token, slice_start, slice_end, limit),
        )
        return [{"doc_id": r[0], "title": r[1]} for r in cur.fetchall()]


def fetch_kwic_for_doc(conn, token: str, doc_id: str, limit: int = 3) -> list[Tuple[str, str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token_idx
            FROM tokens
            WHERE doc_id = %s
              AND token = %s
            LIMIT %s;
            """,
            (doc_id, token, limit),
        )
        positions = [r[0] for r in cur.fetchall()]

    rows = []
    for idx in positions:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT token_idx, token
                FROM tokens
                WHERE doc_id = %s
                  AND token_idx BETWEEN %s AND %s
                ORDER BY token_idx;
                """,
                (doc_id, idx - CONTEXT_WINDOW, idx + CONTEXT_WINDOW),
            )
            context = cur.fetchall()

        left = " ".join(tok for i, tok in context if i < idx)
        kw = next(tok for i, tok in context if i == idx)
        right = " ".join(tok for i, tok in context if i > idx)
        rows.append((left, kw, right))

    return rows


# HTML
def make_html_row(left, kw, right, sim, freq, title, url):
    return f"""
    <tr>
        <td class="left">{left}</td>
        <td class="kw">{kw}</td>
        <td class="right">{right}</td>
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
    td, th { border: 1px solid #ccc; padding: 4px; }
    .kw { font-weight: bold; text-align: center; background: #ffe; }
    .left { text-align: right; color: #666; }
    .right { text-align: left; color: #666; }
    </style></head><body>
    <h1>Concept Neighbour KWIC Audit</h1>
    """]

    for slice_key, concepts in audit_data.items():
        parts.append(f"<h2>Slice {slice_key}</h2>")

        for concept, seeds in concepts.items():
            parts.append(f"<h3>Concept: {concept}</h3>")

            for seed, neighbours in seeds.items():
                parts.append(f"<h4>Seed: {seed}</h4>")
                parts.append("""
                <table>
                <tr>
                    <th>Left</th><th>Keyword</th><th>Right</th>
                    <th>Sim</th><th>Freq</th><th>Doc</th>
                </tr>
                """)

                for n in neighbours:
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
    logger.info("Starting concept neighbour explorer")

    last_log_time = time.time()
    audit: Dict[str, Any] = {}

    with eebo_db.get_connection() as conn:
        for slice_range in config.SLICES:
            slice_start, slice_end = slice_range
            slice_key = f"{slice_start}_{slice_end}"
            index, vocab = load_slice_index(slice_range)
            audit[slice_key] = {}

            # CACHE ALL SEED VECTORS FOR THIS SLICE
            seed_vectors: dict[str, np.ndarray] = {}
            for _concept, cfg in config.CONCEPT_SETS.items():
                for seed in cfg["forms"]:
                    vec = get_vector(conn, seed, slice_start, slice_end)
                    if vec is not None:
                        seed_vectors[seed] = vec

            for concept, cfg in config.CONCEPT_SETS.items():
                audit[slice_key][concept] = {}

                for seed in cfg["forms"]:
                    vec = seed_vectors.get(seed)
                    if vec is None:
                        continue

                    now = time.time()
                    if now - last_log_time > LOG_INTERVAL:
                        logger.info(
                            f"Processing slice {slice_key}, concept {concept}, seed {seed}"
                        )
                        last_log_time = now

                    # FAISS SEARCH
                    D, Idx = index.search(vec.reshape(1, -1), TOP_K)
                    top_neighbors = [
                        (vocab[idx], float(sim))
                        for sim, idx in zip(D[0], Idx[0], strict=True)
                        if sim >= SIM_THRESHOLD and vocab[idx] != seed
                    ]

                    if not top_neighbors:
                        continue

                    # BATCH FETCH FREQUENCY
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

                    # BATCH FETCH DOCUMENTS
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
                        docs_map: dict[str, list[dict[str, str]]] = {}
                        for token_, doc_id, title in cur.fetchall():
                            docs_map.setdefault(token_, []).append({"doc_id": doc_id, "title": title})

                    # FETCH KWIC PER TOKEN (PER DOC)
                    neighbors_list = []
                    for token, sim in top_neighbors:
                        token_freq = freq_map.get(token, 0)

                        # docs: list of dicts, each dict has "doc_id": str, "title": str, "kwic": list[tuple[str,str,str]]
                        docs: List[Dict[str, Any]] = docs_map.get(token, [])

                        for d in docs:
                            d["kwic"] = fetch_kwic_for_doc(conn, token, d["doc_id"])

                        neighbors_list.append({
                            "token": token,
                            "similarity": sim,
                            "frequency": token_freq,
                            "documents": docs
                        })

                    audit[slice_key][concept][seed] = neighbors_list

    # Write JSON
    json_path = config.OUT_DIR / "concept_neighbour_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote {json_path}")

    # Write HTML
    html = build_html(audit)
    html_path = config.OUT_DIR / "concept_kwic_audit.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {html_path}")

    logger.info("Explorer complete.")


if __name__ == "__main__":
    main()
