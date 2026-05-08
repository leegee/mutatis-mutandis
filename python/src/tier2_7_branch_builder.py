#!/usr/bin/env python
"""
tier2_7_branch_builder.py - Tier 2.7: k-Branch Persistence Model for Semantic Continuity

This stage constructs a temporally constrained, branching continuity graph
over precomputed Tier 2 cluster structures without introducing new embeddings
or performing any re-clustering.

Core idea:
    Semantic change is modelled as *structured continuity under uncertainty*,
    not as a single lineage or centroid trajectory.

Key mechanisms:

    1. Candidate edge construction
        - Cluster centroids are compared between adjacent temporal slices
        - Cosine similarity is used as the primary continuity signal
        - Additional penalties are applied for:
            * cluster size divergence
            * temporal distance between slices

    2. K-branch retention
        - Each (slice, cluster) node retains up to k highest-scoring outgoing
          continuations
        - This preserves ambiguity in semantic evolution rather than forcing
          deterministic successor chains

    3. Node identity model
        - Nodes are uniquely identified by (slice, cluster)
        - No global re-indexing or semantic renormalisation is performed

    4. Diagnostic stability analysis (non-binding)
        - Measures entropy of successor distributions per node
        - Computes dominance ratios for continuation pathways
        - Flags stable vs unstable cluster persistence patterns

Outputs:

    Graph structure:
        {
            nodes: [
                { slice, cluster, size, id }
            ],
            links: [
                { source, target, similarity, score }
            ],
            k: int
        }

    Diagnostics:
        - cluster stability reports (entropy-based)
        - continuation ambiguity measures

Invariants:

    - No embeddings are recomputed
    - No clustering is performed at this stage
    - All structure is derived from Tier 2 outputs
    - Output represents continuity topology, not semantic truth
"""

from __future__ import annotations

import json
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np

from lib.eebo_config import ZARR_ROOT
from lib.eebo_logging import logger
from tier2_0_concept_structure import OUTPUT_PATH as INPUT_PATH

# OUTPUT_PATH = ZARR_ROOT / "tier2" / "tier2_7_kbranch.json"
OUTPUT_PATH = ZARR_ROOT / "tier2" / "d3_export.json" # Temp


def load_structure() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload["data"]["tokens"]


def node_id(slice_id: str, cluster_id: int) -> str:
    return f"{slice_id}:{cluster_id}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """
    Stable cosine similarity.

    EPS smoothing prevents division instability for sparse vectors.
    """

    denom = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    ) + 1e-12

    return float(np.dot(a, b) / denom)


def temporal_gap(a: str, b: str) -> int:
    """
    Structural temporal distance between slices.

    Uses slice start-year only.
    """

    ay = int(a.split("-")[0])
    by = int(b.split("-")[0])

    return abs(by - ay)


def score_edge(edge: Dict[str, Any]) -> float:
    """
    Structural continuity score.

    Balances:
        - semantic similarity
        - cluster size stability
        - temporal adjacency

    Similarity dominates.
    """

    sim = float(edge["similarity"])

    size_penalty = abs(
        edge["from_size"] - edge["to_size"]
    ) / max(edge["from_size"], 1)

    gap_penalty = 0.05 * temporal_gap(
        edge["from_slice"],
        edge["to_slice"]
    )

    return sim - (0.15 * size_penalty) - gap_penalty


# continuity construction

def build_candidate_edges(
    token_data: Dict[str, Any],
    similarity_threshold: float = 0.75
):
    """
    Build candidate continuity edges between adjacent slices.

    IMPORTANT:
        Cluster IDs are NOT assumed stable across slices.

    Continuity is inferred entirely from centroid similarity.
    """

    slice_ids = sorted(token_data.keys())

    edges = []

    for i in range(len(slice_ids) - 1):
        s1 = slice_ids[i]
        s2 = slice_ids[i + 1]

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

                    "similarity": float(sim)
                })

    logger.info(
        f"[tier2.7-edge] {token_data.keys()} "
        f"size_range=({min(a['size'] for s in token_data.values() for a in s.get('clusters', []) if s.get('clusters'))}, "
        f"{max(a['size'] for s in token_data.values() for a in s.get('clusters', []) if s.get('clusters'))})"
    )
    return edges


# k-branch graph construction

def group_by_source(edges: List[Dict[str, Any]]):
    by_src = defaultdict(list)

    for e in edges:
        key = (e["from_slice"], e["from_cluster"])
        by_src[key].append(e)

    return by_src


def build_k_branch_graph(edges: List[Dict[str, Any]], k: int = 3):

    by_src = group_by_source(edges)

    node_mass = defaultdict(int)

    links = []

    retained_edges = []

    for src, outgoing in by_src.items():

        ranked = sorted(outgoing, key=score_edge, reverse=True)

        retained = [
            e for e in ranked[:k]
            if score_edge(e) > 0.6
        ]

        retained_edges.extend(retained)

    # FIRST PASS: compute mass globally
    for e in retained_edges:
        src_node = (e["from_slice"], e["from_cluster"])
        dst_node = (e["to_slice"], e["to_cluster"])

        node_mass[src_node] += e["from_size"]
        node_mass[dst_node] += e["to_size"]

    nodes = {}

    for node, mass in node_mass.items():
        nodes[node] = {
            "id": f"{node[0]}:{node[1]}",
            "slice": node[0],
            "cluster": node[1],
            "size": mass
        }

    # SECOND PASS: links only reference stable IDs
    for e in retained_edges:
        src_node = (e["from_slice"], e["from_cluster"])
        dst_node = (e["to_slice"], e["to_cluster"])

        links.append({
            "source": f"{src_node[0]}:{src_node[1]}",
            "target": f"{dst_node[0]}:{dst_node[1]}",
            "similarity": e["similarity"],
            "score": score_edge(e),
            "from_size": e["from_size"],
            "to_size": e["to_size"],
            "from_slice": e["from_slice"],
            "to_slice": e["to_slice"]
        })

    return {
        "nodes": list(nodes.values()),
        "links": links,
        "k": k
    }


def compute_cluster_stability_report(
    edges: List[Dict[str, Any]]
):
    """
    Diagnostic only.

    Measures whether clusters behave like persistent
    semantic entities across slices.

    Low entropy:
        stable continuation

    High entropy:
        fragmentation / semantic divergence / instability
    """

    outgoing = defaultdict(list)
    incoming = defaultdict(list)

    for e in edges:

        src = (
            e["from_slice"],
            e["from_cluster"]
        )

        dst = (
            e["to_slice"],
            e["to_cluster"]
        )

        sim = float(e["similarity"])

        outgoing[src].append((dst, sim))
        incoming[dst].append((src, sim))

    reports = []

    for src, outs in outgoing.items():

        if not outs:
            continue

        # aggregate by destination cluster
        agg = defaultdict(float)

        for (dst, sim) in outs:
            dst_cluster = dst[1]
            agg[dst_cluster] += sim

        weights = np.asarray(
            list(agg.values()),
            dtype=float
        )

        if len(weights) == 0:
            continue

        weights = weights / (
            weights.sum() + 1e-12
        )

        entropy = -np.sum(
            weights * np.log(weights + 1e-12)
        )

        dominant_ratio = float(np.max(weights))

        reports.append({
            "source_slice": src[0],
            "source_cluster": src[1],

            "out_degree": len(outs),

            # lower = more stable
            "entropy": float(entropy),

            # higher = more stable
            "dominant_successor_ratio": dominant_ratio,

            # heuristic interpretability flag
            "is_stable": bool(
                entropy < 0.7 and
                dominant_ratio > 0.6
            )
        })

    return reports


def build_all():

    data = load_structure()

    graphs = {}
    diagnostics = {}

    for token, token_data in data.items():
        logger.info(f"[tier2.7] token={token}")

        edges = build_candidate_edges(token_data)

        if not edges:
            logger.info(
                f"[tier2.7] skip token={token} (no edges)"
            )
            continue

        graph = build_k_branch_graph(edges, k=3)
        stability = compute_cluster_stability_report(edges)
        graphs[token] = graph

        diagnostics[token] = { "cluster_stability": stability }

        logger.info(
            f"[tier2.7] {token}: "
            f"nodes={len(graph['nodes'])}, "
            f"links={len(graph['links'])}"
        )

    return graphs, diagnostics


def write_output(
    graphs: Dict[str, Any],
    diagnostics: Dict[str, Any]
):

    payload = {
        "data": graphs,
        "diagnostics": diagnostics
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[tier2.7] wrote {OUTPUT_PATH}")


def main():
    graphs, diagnostics = build_all()

    write_output(
        graphs=graphs,
        diagnostics=diagnostics
    )


if __name__ == "__main__":
    main()
