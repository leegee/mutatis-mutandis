#!/usr/bin/env python
"""
Merged Concept Neighbour Explorer & Trajectory Visualizer

- Loads aligned slice embeddings (default)
- Computes concept neighbours for target concepts
- Generates JSON audit for external reference
- Builds rank trajectories, network plot, and Gantt chart
"""

from __future__ import annotations
import json
from typing import Any, Dict, List,  TypedDict
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from html import escape

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import faiss

from slice_embedding_pipeline import load_aligned_vectors, search_faiss, add_to_faiss_index
from lib.eebo_logging import logger
from lib.eebo_config import SLICES, CONCEPT_SETS, OUT_DIR, TEXT_BASE_URL
from lib.eebo_db import get_connection


TOP_K = 5
BACKEND='fasttext'
TARGET = "LIBERTY"
KWIC_MAX_LEFT = 40
KWIC_MAX_RIGHT = 40
SIM_THRESHOLD = 0.7
MIN_FREQ = 5
MIN_SIM = 0.5
GANTT_YEAR_SCALE = 0.3

json_path = OUT_DIR / f"concept_neighbour_audit_{TARGET.lower()}.json"

def fetch_kwic_for_token_in_slice(token: str, slice_start: int, slice_end: int) -> List[Dict[str, Any]]:
    """Return KWICs for all docs in slice containing `token`."""
    token_lower = token.lower()
    results = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Get documents in this slice that contain the token
            cur.execute("""
                SELECT DISTINCT d.doc_id, d.title
                FROM pamphlet_tokens t
                JOIN pamphlet_corpus d ON t.doc_id = d.doc_id
                WHERE t.token = %s AND d.slice_start = %s AND d.slice_end = %s
            """, (token_lower, slice_start, slice_end))

            docs = cur.fetchall()
            for doc_id, title in docs:
                # fetch KWIC window
                cur.execute("""
                    SELECT token_idx, token
                    FROM pamphlet_tokens
                    WHERE doc_id = %s
                    AND token = %s
                    ORDER BY token_idx
                """, (doc_id, token_lower))
                positions = [r[0] for r in cur.fetchall()]

                kwic_list = []
                for idx in positions:
                    left_idx = max(0, idx - KWIC_MAX_LEFT)
                    right_idx = idx + KWIC_MAX_RIGHT
                    cur.execute("""
                        SELECT token
                        FROM pamphlet_tokens
                        WHERE doc_id = %s AND token_idx BETWEEN %s AND %s
                        ORDER BY token_idx
                    """, (doc_id, left_idx, right_idx))
                    window = [r[0] for r in cur.fetchall()]
                    left = " ".join(window[:idx - left_idx]) if idx - left_idx > 0 else ""
                    kw = window[idx - left_idx] if idx - left_idx < len(window) else token
                    right = " ".join(window[idx - left_idx + 1:]) if idx - left_idx + 1 < len(window) else ""
                    kwic_list.append((left, kw, right))

                results.append({"doc_id": doc_id, "title": title, "kwic": kwic_list})

    return results


def make_html_row(left, kw, right, sim, freq, title, url):
    left_q = escape(left[-KWIC_MAX_LEFT:])
    kw_q = f"<strong>{escape(kw)}</strong>"
    right_q = escape(right[:KWIC_MAX_RIGHT])
    return f"""
    <tr>
        <td class="left">…{left_q}</td>
        <td class="kw">{kw_q}</td>
        <td class="right">{right_q}…</td>
        <td>{sim:.3f}</td>
        <td>{freq}</td>
        <td><a href="{escape(url)}" target="_blank">{escape(title)}</a></td>
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
                        url = f"{TEXT_BASE_URL}{doc['doc_id']}"
                        parts.append(make_html_row(left, kw, right, n["similarity"], n["frequency"], doc["title"], url))
            parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)

# TypedDict for trajectory entries
class TrajectoryEntry(TypedDict):
    slice: str
    rank: int
    frequency: int
    avg_similarity: float

# Data class for neighbour statistics
@dataclass
class NeighbourStats:
    similarities: List[float] = field(default_factory=list)
    total_frequency: int = 0
    occurrences: int = 0

# Main exploration & plotting
def main():
    logger.info("Starting concept neighbour exploration")

    audit: Dict[str, Any] = {}

    for slice_range in SLICES:
        slice_id = f"{slice_range[0]}_{slice_range[1]}"
        logger.info(f"Processing slice {slice_id}")

        # Load aligned embeddings
        vectors = load_aligned_vectors(f"{slice_range[0]}-{slice_range[1]}", BACKEND)
        words = list(vectors.keys())
        dim = next(iter(vectors.values())).shape[0]

        # Build FAISS index
        index = faiss.IndexFlatIP(dim)
        vec_matrix = np.stack([vectors[w] / np.linalg.norm(vectors[w]) for w in words])
        add_to_faiss_index(index, vec_matrix)

        audit[slice_id] = {}

        for concept in CONCEPT_SETS.keys():
            if concept != TARGET.upper():
                continue
            seed = concept.lower()
            if seed not in vectors:
                logger.warning(f"No vector for concept {seed} in slice {slice_id}")
                continue
            seed_vec = vectors[seed] / np.linalg.norm(vectors[seed])
            D, _I = search_faiss(index, seed_vec.reshape(1, -1), TOP_K)

            neighbours_list = []
            for sim, idx in zip(D[0], _I[0], strict=True):
                if idx == -1:
                    continue
                token = words[idx]
                if token in CONCEPT_SETS[concept].get("forms", set()):
                    continue
                if sim < SIM_THRESHOLD:
                    continue

                docs = fetch_kwic_for_token_in_slice(token, slice_range[0], slice_range[1])
                freq = sum(len(d["kwic"]) for d in docs)
                if freq < MIN_FREQ:
                    continue
                neighbours_list.append({
                    "token": token,
                    "similarity": float(sim),
                    "frequency": freq,
                    "documents": docs
                })

            audit[slice_id][concept] = {"probe": seed, "neighbours": neighbours_list}

    # Write JSON audit
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote audit JSON to {json_path}")

    # Build HTML
    html = build_html(audit)
    html_path = OUT_DIR / f"concept_kwic_audit_{TARGET.lower()}.html"
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote KWIC HTML to {html_path}")

    # Build token trajectories
    summary_by_slice: Dict[str, List[Dict[str, Any]]] = {}
    token_trajectories: Dict[str, List[TrajectoryEntry]] = defaultdict(list)

    for slice_name, probes in audit.items():
        neighbour_data: defaultdict[str, NeighbourStats] = defaultdict(NeighbourStats)
        for probe_info in probes.values():
            for n in probe_info.get("neighbours", []):
                if n["frequency"] < MIN_FREQ or n["similarity"] < MIN_SIM:
                    continue
                token = n["token"]
                neighbour_data[token].similarities.append(n["similarity"])
                neighbour_data[token].total_frequency += n["frequency"]
                neighbour_data[token].occurrences += 1

        slice_summary: List[Dict[str, Any]] = []
        for token, stats in neighbour_data.items():
            slice_summary.append({
                "token": token,
                "avg_similarity": mean(stats.similarities),
                "total_frequency": stats.total_frequency,
                "times_as_neighbour": stats.occurrences
            })

        slice_summary.sort(key=lambda x: x["total_frequency"], reverse=True)

        for rank, entry in enumerate(slice_summary, start=1):
            te: TrajectoryEntry = TrajectoryEntry(
                slice=slice_name,
                rank=rank,
                frequency=entry["total_frequency"],
                avg_similarity=entry["avg_similarity"]
            )
            token_trajectories[entry["token"]].append(te)

        summary_by_slice[slice_name] = slice_summary

    # Plot rank trajectories
    plt.figure(figsize=(12, 6))
    for token, trajectory in token_trajectories.items():
        xs = [int(e["slice"].split("_")[0]) + (int(e["slice"].split("_")[1]) - int(e["slice"].split("_")[0])) // 2 for e in trajectory]
        ys = [e["rank"] for e in trajectory]
        sorted_pairs = sorted(zip(xs, ys, strict=True), key=lambda p: p[0])
        xs_sorted, ys_sorted = map(list, zip(*sorted_pairs, strict=True))
        plt.plot(xs_sorted, ys_sorted, marker="o", label=token)
    plt.gca().invert_yaxis()
    plt.xlabel("Year (mid-slice)")
    plt.ylabel("Neighbour rank (1 = highest frequency)")
    plt.title("Neighbour-rank trajectories")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot network
    G = nx.DiGraph()
    slice_names = sorted(summary_by_slice.keys())
    slice_to_col = {s: i for i, s in enumerate(slice_names)}
    for token, trajectory in token_trajectories.items():
        if all(entry["frequency"] >= MIN_FREQ for entry in trajectory):
            cols = [slice_to_col[entry["slice"]] for entry in trajectory]
            ranks = [entry["rank"] for entry in trajectory]
            for i in range(len(trajectory)):
                node_id = f"{token}_{cols[i]}"
                G.add_node(node_id, label=token, col=cols[i], rank=ranks[i])
                if i > 0:
                    prev_node_id = f"{token}_{cols[i-1]}"
                    G.add_edge(prev_node_id, node_id)

    pos = {}
    col_to_nodes = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        col_to_nodes[attrs["col"]].append(node_id)
    for col, nodes_in_col in col_to_nodes.items():
        n = len(nodes_in_col)
        max_label_len = max(len(node.split("_")[0]) for node in nodes_in_col)
        for i, node_id in enumerate(nodes_in_col):
            offset = (i - (n-1)/2) * max_label_len * 0.25
            pos[node_id] = (col + offset, -float(G.nodes[node_id]["rank"]))

    plt.figure(figsize=(max(12, len(slice_names)*1.5), 6))
    nx.draw(G, pos, with_labels=False, node_size=600, node_color="#ddd", edgecolors="#555")
    for node_id, (x, y) in pos.items():
        plt.text(x, y, G.nodes[node_id]["label"], fontsize=9, ha="center", va="bottom",
                 bbox=dict(facecolor="white", edgecolor="none", pad=1))
    plt.xlabel("Year slices")
    plt.ylabel("Neighbour rank (1 = highest frequency)")
    plt.title("Neighbour trajectories as network nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gantt chart
    slice_starts = [int(s.split("_")[0]) for s in slice_names]
    slice_ends = [int(s.split("_")[1]) for s in slice_names]
    slice_widths = [end-start+1 for start, end in zip(slice_starts, slice_ends, strict=True)]
    tokens = sorted(token_trajectories.keys())
    num_tokens = len(tokens)
    gantt_data = defaultdict(list)
    for token in tokens:
        trajectory_slices = {entry["slice"]: entry for entry in token_trajectories[token]}
        for slice_name, start, width in zip(slice_names, slice_starts, slice_widths, strict=True):
            present = slice_name in trajectory_slices and trajectory_slices[slice_name]["frequency"] >= MIN_FREQ
            gantt_data[token].append((start, width, present))

    fig, ax = plt.subplots(figsize=(max(24, len(slice_names)*0.6), max(0.5*num_tokens, 6)))
    bar_height = 0.8
    for i, token in enumerate(tokens):
        for start, width, present in gantt_data[token]:
            scaled_start = start * GANTT_YEAR_SCALE
            scaled_width = width * GANTT_YEAR_SCALE
            facecolor = 'skyblue' if present else 'none'
            edgecolor = 'black' if present else 'lightgray'
            linestyle = 'solid' if present else 'dashed'
            ax.broken_barh([(scaled_start, scaled_width)], (i-bar_height/2, bar_height),
                           facecolors=facecolor, edgecolors=edgecolor, linestyle=linestyle)

    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens, fontsize=12)
    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Tokens", fontsize=14)
    ax.set_title("EEBO token trajectories in non-Latin shorter texts", fontsize=16)
    xticks = [int(s.split("_")[0]) for s in slice_names]
    ax.set_xticks([x * GANTT_YEAR_SCALE for x in xticks])
    ax.set_xticklabels([str(x) for x in xticks], fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    svg_file = OUT_DIR / f"concept_trajectory_{TARGET.lower()}.svg"
    plt.savefig(svg_file, format='svg')
    logger.info(f"Gantt chart SVG written to {svg_file}")
    plt.show()

if __name__ == "__main__":
    main()
