#!/usr/bin/env python
"""
tier2_7_branch_builder.py

k-Branch persistence layer.

Key property:
    Nodes are NOT semantic identities.
    They are slice-local cluster observations linked via centroid similarity.

Invariant:
    - doc weights accumulate across retained edges
    - normalisation only occurs at final node materialisation
    - cluster IDs are local-only labels (NOT stable identities)
"""

from __future__ import annotations

import json
from typing import Dict, Any
from collections import defaultdict

import numpy as np

from lib.eebo_config import OUT_DIR
from lib.eebo_logging import logger

from tier2_0_initial_structures import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = OUT_DIR / "tier2_7.json"

log = logger


# -----------------------------
# IO
# -----------------------------

def load_structure() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["data"]["tokens"]


# -----------------------------
# Geometry
# -----------------------------

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


# -----------------------------
# Edge construction
# -----------------------------

def _safe_centroid(c: Dict[str, Any]) -> np.ndarray | None:
    if c is None:
        return None
    centroid = c.get("centroid", None)
    if centroid is None:
        return None
    return np.asarray(centroid, dtype=np.float32)


def build_candidate_edges(token_data: Dict[str, Any], similarity_threshold: float = 0.75):

    slice_ids = sorted(token_data.keys())
    edges = []

    log.info(f"[tier2.7] building_edges slices={len(slice_ids)}")

    for i in range(len(slice_ids) - 1):
        s1, s2 = slice_ids[i], slice_ids[i + 1]

        c1 = token_data[s1].get("clusters", {}).get("centered", [])
        c2 = token_data[s2].get("clusters", {}).get("centered", [])

        log.debug(
            f"[tier2.7] slice_pair={s1}->{s2} "
            f"clusters={len(c1)}->{len(c2)}"
        )

        if not c1 or not c2:
            log.debug(f"[tier2.7] skip_pair empty_clusters {s1}->{s2}")
            continue

        for a in c1:
            va = _safe_centroid(a)
            if va is None:
                continue

            if a.get("degenerate"):
                log.debug(f"[tier2.7] skip_cluster degenerate from {s1}:{a.get('cluster_id')}")
                continue

            for b in c2:
                vb = _safe_centroid(b)
                if vb is None:
                    continue

                if b.get("degenerate"):
                    continue

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

                    "from_docs": a.get("doc_weights", {}),
                    "to_docs": b.get("doc_weights", {})
                })

        log.debug(
            f"[tier2.7] slice_pair_done={s1}->{s2} edges_so_far={len(edges)}"
        )

    log.info(f"[tier2.7] total_edges={len(edges)}")

    return edges


# -----------------------------
# Edge grouping
# -----------------------------

def group_by_source(edges):
    by_src = defaultdict(list)
    for e in edges:
        by_src[(e["from_slice"], e["from_cluster"])].append(e)
    return by_src


# -----------------------------
# Graph construction
# -----------------------------

def build_k_branch_graph(edges, k=3):

    by_src = group_by_source(edges)

    log.info(
        f"[tier2.7] k_branch_start edges={len(edges)} "
        f"groups={len(by_src)}"
    )

    retained_edges = []

    def score(e):
        return score_edge(e)

    for src, outgoing in by_src.items():
        ranked = sorted(outgoing, key=score, reverse=True)

        kept = 0

        for e in ranked[:k]:
            s = score_edge(e)
            if s > 0.6:
                retained_edges.append(e)
                kept += 1

        if kept == 0:
            log.debug(f"[tier2.7] prune_node {src} no_retained_edges")

    node_intrinsic = defaultdict(int)
    node_doc_weights = defaultdict(lambda: defaultdict(float))

    # -----------------------------
    # ACCUMULATION
    # -----------------------------
    for e in retained_edges:

        src = (e["from_slice"], e["from_cluster"])
        dst = (e["to_slice"], e["to_cluster"])

        node_intrinsic[src] = max(node_intrinsic[src], e["from_size"])
        node_intrinsic[dst] = max(node_intrinsic[dst], e["to_size"])

        for doc, w in e["from_docs"].items():
            node_doc_weights[src][doc] += float(w)

        for doc, w in e["to_docs"].items():
            node_doc_weights[dst][doc] += float(w)

    log.info(
        f"[tier2.7] retained_edges={len(retained_edges)} "
        f"nodes_src={len(node_intrinsic)}"
    )

    # -----------------------------
    # NORMALISATION
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


# -----------------------------
# Driver
# -----------------------------

def build_all():
    data = load_structure()

    graphs = {}
    diagnostics = {}

    for token, token_data in data.items():
        log.info(f"[tier2.7] token={token} slices={len(token_data)}")

        edges = build_candidate_edges(token_data)

        log.info(f"[tier2.7] token={token} edges={len(edges)}")

        if not edges:
            log.warning(f"[tier2.7] DROP_TOKEN_NO_EDGES token={token}")
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

    log.info(f"[tier2.7] wrote {OUTPUT_PATH}")


def main():
    graphs, diagnostics = build_all()
    write_output(graphs, diagnostics)


if __name__ == "__main__":
    main()
