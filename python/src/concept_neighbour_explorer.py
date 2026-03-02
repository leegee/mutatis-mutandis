#!/usr/bin/env python
"""
concept_neighbour_explorer.py

Exploratory concept neighbour & KWIC audit using vectors from slice_embedding_pipeline.

Defaults to aligned vectors (USE_ALIGNED_VECTORS=True).

Also includes:
- neighbour statistics per slice
- rank trajectories
- network visualization
- Gantt chart of token presence across slices
"""

from __future__ import annotations
import os
import json
from typing import Any, Dict, List
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from lib.eebo_config import SLICES, CONCEPT_SETS, OUT_DIR
from lib.eebo_db import get_connection
from lib.eebo_logging import logger
from lib.faiss_slices import load_slice_index
from slice_embedding_pipeline import load_aligned_vectors, load_unaligned_vectors

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
def fetch_kwic_for_doc(conn, token: str, doc_id: str, limit: int = 3) -> List[tuple[str,str,str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT token_idx FROM pamphlet_tokens WHERE doc_id = %s AND token = %s LIMIT %s",
            (doc_id, token, limit)
        )
        positions = [r[0] for r in cur.fetchall()]

    rows: List[tuple[str,str,str]] = []
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
            (doc_id, min_idx, max_idx)
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
                        parts.append(make_html_row(left, kw, right, n["similarity"], n["frequency"], doc["title"], url))
            parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts)

# --- Main concept neighbour audit ---
def run_audit() -> None:
    logger.info("Starting concept neighbour explorer (aligned=%s)", USE_ALIGNED_VECTORS)
    audit: Dict[str, Any] = {}

    with get_connection() as conn:
        for slice_range in SLICES:
            slice_start, slice_end = slice_range
            slice_key = f"{slice_start}_{slice_end}"

            index, vocab = load_slice_index(slice_range, use_aligned=USE_ALIGNED_VECTORS)

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
                D, Idx = index.search(vec.reshape(1, -1), TOP_K)
                top_neighbors = [(vocab[idx], float(sim)) for sim, idx in zip(D[0], Idx[0], strict=True) if idx != -1]

                known_forms = meta.get("forms", set())
                false_positives = meta.get("false_positives", set())
                top_neighbors = [(tok, sim) for tok, sim in top_neighbors if tok not in known_forms and tok not in false_positives]

                if not top_neighbors:
                    audit[slice_key][concept] = {"probe": seed, "neighbours": []}
                    continue

                tokens = [t for t, _ in top_neighbors]

                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT token, COUNT(*) FROM pamphlet_tokens WHERE token = ANY(%s) AND slice_start = %s AND slice_end = %s GROUP BY token",
                        (tokens, slice_start, slice_end)
                    )
                    freq_map = {r[0]: r[1] for r in cur.fetchall()}

                    cur.execute(
                        "SELECT DISTINCT token, doc_id, title FROM pamphlet_tokens WHERE token = ANY(%s) AND slice_start = %s AND slice_end = %s",
                        (tokens, slice_start, slice_end)
                    )
                    docs_map: Dict[str, List[Dict[str,str]]] = {}
                    for token_, doc_id, title in cur.fetchall():
                        docs_map.setdefault(token_, []).append({"doc_id": doc_id, "title": title})

                neighbours_list: List[Dict[str, Any]] = []
                for token, sim in top_neighbors:
                    token_freq = freq_map.get(token, 0)
                    docs: List[Dict[str, Any]] = docs_map.get(token, [])
                    for d in docs:
                        d["kwic"] = fetch_kwic_for_doc(conn, token, d["doc_id"])
                    neighbours_list.append({"token": token, "similarity": sim, "frequency": token_freq, "documents": docs})

                audit[slice_key][concept] = {"probe": seed, "neighbours": neighbours_list}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Wrote {json_path}")

    html = build_html(audit)
    html_path.write_text(html, encoding="utf-8")
    logger.info(f"Wrote {html_path}")

# --- Neighbour statistics & visualizations ---
@dataclass
class NeighbourStats:
    similarities: List[float] = field(default_factory=list)
    total_frequency: int = 0
    occurrences: int = 0

def analyze_and_plot() -> None:
    INPUT_FILE = json_path
    SVG_FILE = INPUT_FILE.with_suffix(".svg")
    MIN_FREQ = 5
    MIN_SIM = 0.5
    GANTT_YEAR_SCALE = 0.3

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Dict]] = json.load(f)

    summary_by_slice: Dict[str, List[Dict]] = {}
    token_trajectories: Dict[str, List[Dict]] = defaultdict(list)

    # --- Build summary per slice ---
    for slice_name, probes in data.items():
        neighbour_data: defaultdict[str, NeighbourStats] = defaultdict(NeighbourStats)
        for _probe_name, probe_info in probes.items():
            for n in probe_info.get("neighbours", []):
                sim = float(n["similarity"])
                freq = int(n.get("frequency",0))
                if freq < MIN_FREQ or sim < MIN_SIM:
                    continue
                token: str = n["token"]
                neighbour_data[token].similarities.append(sim)
                neighbour_data[token].total_frequency += freq
                neighbour_data[token].occurrences += 1

        slice_summary: List[Dict] = []
        for token, stats in neighbour_data.items():
            slice_summary.append({
                "token": token,
                "avg_similarity": mean(stats.similarities),
                "total_frequency": stats.total_frequency,
                "times_as_neighbour": stats.occurrences
            })
        slice_summary.sort(key=lambda x: x["total_frequency"], reverse=True)
        for rank, entry in enumerate(slice_summary, start=1):
            token_trajectories[entry["token"]].append({
                "slice": slice_name,
                "rank": rank,
                "frequency": entry["total_frequency"],
                "avg_similarity": entry["avg_similarity"]
            })
        summary_by_slice[slice_name] = slice_summary

    # --- Log summary ---
    for slice_name, neighbours in summary_by_slice.items():
        logger.info(f"\n## SLICE {slice_name} ##")
        for n in neighbours:
            if n["total_frequency"] >= MIN_FREQ:
                logger.info(f"{n['token']:15} freq={n['total_frequency']:5} avg_sim={n['avg_similarity']:.3f} seen={n['times_as_neighbour']}")

    # --- Trajectory plot ---
    plt.figure(figsize=(12,6))
    for token, trajectory in token_trajectories.items():
        xs = [int(entry["slice"].split("_")[0]) + (int(entry["slice"].split("_")[1])-int(entry["slice"].split("_")[0]))//2 for entry in trajectory]
        ys = [entry["rank"] for entry in trajectory]
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

    # --- Network plot ---
    G = nx.DiGraph()
    slice_names = sorted(summary_by_slice.keys())
    slice_to_col = {s:i for i,s in enumerate(slice_names)}
    for token, trajectory in token_trajectories.items():
        if all(entry["frequency"] >= MIN_FREQ for entry in trajectory):
            cols = [slice_to_col[entry["slice"]] for entry in trajectory]
            ranks = [entry["rank"] for entry in trajectory]
            for i in range(len(trajectory)):
                node_id = f"{token}_{cols[i]}"
                G.add_node(node_id, label=token, col=cols[i], rank=ranks[i], slice=trajectory[i]["slice"])
                if i>0:
                    prev_node_id = f"{token}_{cols[i-1]}"
                    G.add_edge(prev_node_id, node_id)
    max_label_len = max(len(n.split("_")[0]) for n in G.nodes())
    col_to_nodes = defaultdict(list)
    for node_id, attrs in G.nodes(data=True):
        col_to_nodes[attrs["col"]].append(node_id)
    pos = {}
    for col, nodes_in_col in col_to_nodes.items():
        n = len(nodes_in_col)
        for i, node_id in enumerate(nodes_in_col):
            offset = (i - (n-1)/2) * max_label_len * 0.25
            pos[node_id] = (col + offset, -float(G.nodes[node_id]["rank"]))
    plt.figure(figsize=(max(12,len(slice_names)*1.5),6))
    nx.draw(G,pos,with_labels=False,node_size=600,node_color="#ddd",edgecolors="#555")
    for node_id,(x,y) in pos.items():
        plt.text(x, y, G.nodes[node_id]["label"], fontsize=9, ha="center", va="bottom", bbox=dict(facecolor="white",edgecolor="none",pad=1))
    plt.xlabel("Year slices")
    plt.ylabel("Neighbour rank (1 = highest frequency)")
    plt.title("Neighbour trajectories as network nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Gantt chart ---
    slice_starts = [int(s.split("_")[0]) for s in slice_names]
    slice_ends   = [int(s.split("_")[1]) for s in slice_names]
    slice_widths = [end-start+1 for start,end in zip(slice_starts,slice_ends, strict=True)]
    tokens = sorted(token_trajectories.keys())
    num_tokens = len(tokens)
    gantt_data: Dict[str,List[tuple]] = defaultdict(list)
    for token in tokens:
        trajectory_slices = {entry["slice"]: entry for entry in token_trajectories[token]}
        for slice_name, start, width in zip(slice_names, slice_starts, slice_widths, strict=True):
            if slice_name in trajectory_slices and trajectory_slices[slice_name]["frequency"]>=MIN_FREQ:
                gantt_data[token].append((start,width,True))
            else:
                gantt_data[token].append((start,width,False))
    fig, ax = plt.subplots(figsize=(max(24,len(slice_names)*0.6),max(0.5*num_tokens,6)))
    bar_height=0.8
    for i, token in enumerate(tokens):
        for start,width,present in gantt_data[token]:
            scaled_start = start * GANTT_YEAR_SCALE
            scaled_width = width * GANTT_YEAR_SCALE
            if present:
                ax.broken_barh([(scaled_start, scaled_width)],(i-bar_height/2,bar_height),facecolors='skyblue',edgecolors='black')
            else:
                ax.broken_barh([(scaled_start, scaled_width)],(i-bar_height/2,bar_height),facecolors='none',edgecolors='lightgray',linestyle='dashed')
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens,fontsize=12)
    ax.set_xlabel("Year",fontsize=14)
    ax.set_ylabel("Tokens",fontsize=14)
    ax.set_title("EEBO token trajectories in non-Latin shorter texts",fontsize=16)
    xticks = [int(s.split("_")[0]) for s in slice_names]
    ax.set_xticks([x*GANTT_YEAR_SCALE for x in xticks])
    ax.set_xticklabels([str(x) for x in xticks],fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(SVG_FILE, format='svg')
    logger.info(f"Gantt chart SVG written to {SVG_FILE}")
    plt.show()

# --- Run ---
if __name__=="__main__":
    run_audit()
    analyze_and_plot()
