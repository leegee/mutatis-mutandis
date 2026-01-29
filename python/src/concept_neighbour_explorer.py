#!/usr/bin/env python
"""
concept_neighbour_explorer.py

Unified concept neighbour audit + KWIC visualisation.

Outputs:
- concept_neighbour_audit.json
- concept_kwic_audit.html
"""

from __future__ import annotations
import json
from typing import Dict, Any

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index, get_vector

TOP_K = 25
SIM_THRESHOLD = 0.75
CONTEXT_WINDOW = 8


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


def fetch_kwic_for_doc(conn, token: str, doc_id: str, limit: int = 3):
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
    body { font-family: monospace; }
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

    audit: Dict[str, Any] = {}

    with eebo_db.get_connection() as conn:
        for slice_range in config.SLICES:
            slice_start, slice_end = slice_range
            slice_key = f"{slice_start}_{slice_end}"
            index, vocab = load_slice_index(slice_range)
            audit[slice_key] = {}

            for concept, cfg in config.CONCEPT_SETS.items():
                audit[slice_key][concept] = {}

                for seed in cfg["forms"]:
                    vec = get_vector(conn, seed, slice_start, slice_end)
                    if vec is None:
                        continue

                    D, Idx = index.search(vec.reshape(1, -1), TOP_K)
                    neighbours = []

                    for sim, idx in zip(D[0], Idx[0], strict=True):
                        token = vocab[idx]
                        if sim < SIM_THRESHOLD or token == seed:
                            continue

                        freq = fetch_token_frequency(conn, token, slice_start, slice_end)
                        docs = fetch_documents_for_token(conn, token, slice_start, slice_end)

                        for d in docs:
                            d["kwic"] = fetch_kwic_for_doc(conn, token, d["doc_id"])

                        neighbours.append({
                            "token": token,
                            "similarity": float(sim),
                            "frequency": freq,
                            "documents": docs
                        })

                    audit[slice_key][concept][seed] = neighbours

    # Write JSON
    json_path = config.OUT_DIR / "concept_neighbour_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # Write HTML
    html = build_html(audit)
    html_path = config.OUT_DIR / "concept_kwic_audit.html"
    html_path.write_text(html, encoding="utf-8")

    logger.info("Explorer complete.")


if __name__ == "__main__":
    main()
