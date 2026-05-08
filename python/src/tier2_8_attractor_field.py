#!/usr/bin/env python
"""
tier2_8_attractor_field.py

Revised model: DISTRIBUTIONAL ATTRACTOR BASINS

Key conceptual shift:
    - NO centroid-as-meaning assumption
    - attractor = weighted distribution over repeated cluster states
    - persistence is structural + temporal, not geometric point stability

Invariant:
    - no embeddings recomputed
    - no clustering
    - no modification of Tier 2.7 output
    - purely statistical + graph-theoretic analysis
"""

from __future__ import annotations

import json
from typing import Dict, Any, Tuple
from collections import defaultdict

from lib.eebo_config import ZARR_ROOT
from lib.eebo_logging import logger
from tier2_7_branch_builder import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = ZARR_ROOT / "tier2" / "tier2_attractors.json"


# smoothing avoids division instability in sparse graphs
EPS = 1e-12

SIM_WEIGHT = 1.0
FREQ_WEIGHT = 1.0
DEPTH_DECAY = 0.85


# ------------------------------------------------------------
# IO
# ------------------------------------------------------------

def load_graph() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# normalisation helpers
# ------------------------------------------------------------

def node_from_edge(endpoint) -> Tuple[str, int]:
    """
    Tier 2.7 edges sometimes encode nodes as:
        - list  ["slice", cluster]
        - tuple ("slice", cluster)
        - or malformed nested structures

    We force canonical form:
        (slice: str, cluster: int)
    """
    if isinstance(endpoint, dict):
        # defensive: should not happen in current pipeline
        return (endpoint["slice"], int(endpoint["cluster"]))

    if isinstance(endpoint, (list, tuple)) and len(endpoint) == 2:
        return (endpoint[0], int(endpoint[1]))

    raise TypeError(f"Invalid node format: {endpoint}")


# ------------------------------------------------------------
# core attractor model
# ------------------------------------------------------------

def build_basin(token_graph: Dict[str, Any]):
    links = token_graph.get("links", [])

    visit_freq = defaultdict(int)
    persistence_mass = defaultdict(float)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    depth_weight = defaultdict(float)

    for e in links:

        # FIX: canonical node extraction
        src = node_from_edge(e["source"])
        dst = node_from_edge(e["target"])

        sim = float(e.get("similarity", 0.0))

        # frequency of structural participation
        visit_freq[src] += 1
        visit_freq[dst] += 1

        # similarity mass accumulation
        persistence_mass[src] += sim
        persistence_mass[dst] += sim

        # topology
        out_degree[src] += 1
        in_degree[dst] += 1

        # graph-depth proxy (NOT time)
        depth_weight[src] += sim * DEPTH_DECAY
        depth_weight[dst] += sim * DEPTH_DECAY


    basins = []

    all_nodes = (
        set(visit_freq.keys()) |
        set(persistence_mass.keys()) |
        set(in_degree.keys()) |
        set(out_degree.keys())
    )

    for node in all_nodes:

        freq = visit_freq[node]
        mass = persistence_mass[node]
        indeg = in_degree[node]
        outdeg = out_degree[node]
        depth = depth_weight[node]

        strength = (
            FREQ_WEIGHT * freq +
            SIM_WEIGHT * mass +
            0.5 * (indeg + outdeg) +
            depth
        )

        basins.append({
            "slice": node[0],
            "cluster": node[1],

            "frequency": int(freq),
            "persistence_mass": float(mass),

            "in_degree": int(indeg),
            "out_degree": int(outdeg),

            "depth_weight": float(depth),

            "strength": float(strength)
        })


    # normalisation is purely visual scaling downstream
    if basins:
        max_strength = max(b["strength"] for b in basins) + EPS

        for b in basins:
            b["strength_norm"] = b["strength"] / max_strength

    return basins


# ------------------------------------------------------------
# pipeline
# ------------------------------------------------------------

def build_all():
    data = load_graph()

    output = {}

    for token, graph in data.items():
        output[token] = {
            "attractor_basins": build_basin(graph)
        }

    return output


def write_output(obj: Dict[str, Any]):
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

    logger.info(f"[tier2.8] wrote {OUTPUT_PATH}")


def main():
    write_output(build_all())


if __name__ == "__main__":
    main()
