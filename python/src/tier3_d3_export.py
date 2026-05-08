#!/usr/bin/env python
"""
tier3_d3_export.py

D3 projection layer:

    - consumes Tier 2.7 k-branch graph
    - produces render-ready nodes + links
    - no enrichment, no inference, no transformation of meaning

Invariant:
    - purely structural transport layer
"""

from __future__ import annotations

import json
import argparse
from typing import Dict, Any

from lib.eebo_config import ZARR_ROOT, CONCEPT_SETS
from lib.eebo_logging import logger

from tier2_7_branch_builder import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = ZARR_ROOT / "tier3" / "d3_export.json"


def load_graph() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def node_id(slice_id: str, cluster_id: int) -> str:
    return f"{slice_id}:{cluster_id}"


def build_d3_graph(token: str, data: Dict[str, Any]) -> Dict[str, Any]:

    graph = data.get(token, {})

    nodes_in = graph.get("nodes", [])
    links_in = graph.get("links", [])

    nodes = {}
    links = []

    # --------------------
    # nodes (pure identity)
    # --------------------
    for n in nodes_in:
        nid = node_id(n["slice"], n["cluster"])

        if nid not in nodes:
            nodes[nid] = {
                "id": nid,
                "slice": n["slice"],
                "cluster": n["cluster"],
                "size": 1
            }

    # --------------------
    # links (structural only)
    # --------------------
    for l in links_in:

        src = node_id(l["source"][0], l["source"][1])
        dst = node_id(l["target"][0], l["target"][1])

        links.append({
            "source": src,
            "target": dst,

            "weight": float(l.get("score", l.get("similarity", 0.0))),
            "similarity": float(l.get("similarity", 0.0)),

            "from_slice": l.get("from_slice"),
            "to_slice": l.get("to_slice")
        })

    # deduplicate
    seen = set()
    clean = []

    for l in links:
        key = (l["source"], l["target"])
        if key in seen:
            continue
        seen.add(key)
        clean.append(l)

    return {
        "token": token,
        "nodes": list(nodes.values()),
        "links": clean
    }


def build_all():
    data = load_graph()

    outputs = {}

    for token in CONCEPT_SETS.keys():
        if token not in data:
            continue

        outputs[token] = build_d3_graph(token, data)

    return outputs


def write(outputs: Dict[str, Any]):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    logger.info(f"[tier3] wrote {OUTPUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=False)
    args = ap.parse_args()

    data = load_graph()

    if args.token:
        write({args.token: build_d3_graph(args.token, data)})
    else:
        write(build_all())


if __name__ == "__main__":
    main()
