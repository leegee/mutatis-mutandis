#!/usr/bin/env python
"""
tier2_7_branch_builder.py

k-Branch persistence layer.

Key property:
    Node = cluster identity + aggregated doc support

Invariant:
    doc weights are accumulated across ALL retained edges
    and normalised only at final node materialisation
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np

from lib.eebo_config import OUT_DIR
from lib.eebo_logging import logger
from tier2_0_concept_structure import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = OUT_DIR / "tier2_7.json"


def load_structure() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]["tokens"]


def node_id(slice_id: str, cluster_id: int) -> str:
    return f"{slice_id}:{cluster_id}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def temporal_gap(a: str, b: str) -> int:
    return abs(int(a.split("-")[0]) - int(b.split("-")[0]))


def score_edge(edge: Dict[str, Any]) -> float:
    sim = float(edge["similarity"])
    size_penalty = abs(edge["from_size"] - edge["to_size"]) / max(edge["from_size"], 1)
    gap_penalty = 0.05 * temporal_gap(edge["from_slice"], edge["to_slice"])
    return sim - (0.15 * size_penalty) - gap_penalty


def build_candidate_edges(token_data: Dict[str, Any], similarity_threshold: float = 0.75):

    slice_ids = sorted(token_data.keys())
    edges = []

    for i in range(len(slice_ids) - 1):
        s1, s2 = slice_ids[i], slice_ids[i + 1]

        c1 = token_data[s1].get("clusters", [])
        c2 = token_data[s2].get("clusters", [])

        if not c1 or not c2:
            continue

        for a in c1:
            va = np.asarray(a["centroid"], dtype=np.float32)

            for b in c2:
                vb = np.asarray(b["centroid"], dtype=np.float32)
                sim = cosine(va, vb)

                if sim < similarity_threshold:
                    continue

                edges.append({
                    "from_slice": s1,
                    "to_slice": s2,
                    "from_cluster": int(a["cluster_id"]),
                    "to_cluster": int(b["cluster_id"]),
                    "from_size": int(a["size"]),
                    "to_size": int(b["size"]),
                    "similarity": float(sim),

                    # raw doc support (may be sparse)
                    "from_docs": a.get("doc_weights", {}),
                    "to_docs": b.get("doc_weights", {})
                })

    return edges


def group_by_source(edges):
    by_src = defaultdict(list)
    for e in edges:
        by_src[(e["from_slice"], e["from_cluster"])].append(e)
    return by_src


def build_k_branch_graph(edges, k=3):

    by_src = group_by_source(edges)
    retained_edges = []

    for _, outgoing in by_src.items():
        ranked = sorted(outgoing, key=score_edge, reverse=True)
        retained_edges.extend([e for e in ranked[:k] if score_edge(e) > 0.6])

    node_intrinsic = defaultdict(int)
    node_doc_weights = defaultdict(lambda: defaultdict(float))

    # -----------------------------
    # ACCUMULATION PHASE
    # -----------------------------
    for e in retained_edges:

        src = (e["from_slice"], e["from_cluster"])
        dst = (e["to_slice"], e["to_cluster"])

        node_intrinsic[src] = max(node_intrinsic[src], e["from_size"])
        node_intrinsic[dst] = max(node_intrinsic[dst], e["to_size"])

        # accumulate doc signal (CRITICAL FIX)
        for doc, w in e["from_docs"].items():
            node_doc_weights[src][doc] += float(w)

        for doc, w in e["to_docs"].items():
            node_doc_weights[dst][doc] += float(w)

    # -----------------------------
    # NORMALISATION PHASE
    # -----------------------------
    nodes = {}

    for node in set(node_intrinsic) | set(node_doc_weights):

        raw = node_doc_weights[node]
        total = sum(raw.values()) + 1e-12

        doc_weights = {k: float(v / total) for k, v in raw.items()}

        nodes[node] = {
            "id": f"{node[0]}:{node[1]}",
            "slice": node[0],
            "cluster": node[1],

            "size": int(node_intrinsic[node]),

            "doc_ids": list(doc_weights.keys()),
            "doc_weights": doc_weights,
            "doc_mass": float(total)
        }

    links = [
        {
            "source": node_id(e["from_slice"], e["from_cluster"]),
            "target": node_id(e["to_slice"], e["to_cluster"]),
            "similarity": e["similarity"],
            "score": score_edge(e),
            "from_slice": e["from_slice"],
            "to_slice": e["to_slice"]
        }
        for e in retained_edges
    ]

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "k": k
    }


def build_all():
    data = load_structure()

    graphs = {}
    diagnostics = {}

    for token, token_data in data.items():
        logger.info(f"[tier2.7] token={token}")

        edges = build_candidate_edges(token_data)
        if not edges:
            continue

        graphs[token] = build_k_branch_graph(edges, k=3)
        diagnostics[token] = {"edges": len(edges)}

    return graphs, diagnostics


def write_output(graphs, diagnostics):
    payload = {
        "data": graphs,
        "diagnostics": diagnostics
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[tier2.7] wrote {OUTPUT_PATH}")


def main():
    graphs, diagnostics = build_all()
    write_output(graphs, diagnostics)


if __name__ == "__main__":
    main()
