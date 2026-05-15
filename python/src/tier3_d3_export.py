#!/usr/bin/env python
"""
tier3_d3_export.py

D3 projection layer:

    - consumes Tier 2.7 k-branch graph
    - attaches document filepaths for UI rendering
    - produces D3-ready JSON (nodes + links)

Invariant:
    - no clustering
    - no similarity computation
    - no semantic transformation
    - only UI binding / enrichment
"""

from __future__ import annotations

import json
import argparse
from typing import Dict, Any
from pathlib import Path

from lib.eebo_db import get_connection
from lib.eebo_config import XML_ROOT_DIR, CONCEPT_SETS, OUT_DIR
from lib.eebo_logging import logger

from tier2_7_branch_builder import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = OUT_DIR / "d3_export.json"



# Shudda done this during ingestion
def normalise_filepath(path: str) -> str:
    if not path:
        return ""

    path = str(path)

    root = str(XML_ROOT_DIR)

    if path.startswith(root):
        rel = path[len(root):]
    else:
        rel = path

    return rel.replace('\\', '/')


def load_doc_map() -> Dict[str, str]:
    """
    doc_id -> filepath lookup table
    """
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT doc_id, filepath
            FROM documents
        """)
        rows = cur.fetchall()

    conn.close()

    return {doc_id: normalise_filepath(filepath) for doc_id, filepath in rows}



def load_graph() -> Dict[str, Any]:
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # Tier 2.7 structure
    return payload.get("data", {})



# HELPERS


def node_id(slice_id: str, cluster_id: int) -> str:
    return f"{slice_id}:{cluster_id}"



# CORE TRANSFORM


def build_d3_graph(
    token: str,
    data: Dict[str, Any],
    doc_map: Dict[str, str]
) -> Dict[str, Any]:

    graph = data.get(token, {})
    nodes_in = graph.get("nodes", [])
    links_in = graph.get("links", [])

    nodes: Dict[str, Any] = {}


    # NODES (enrichment only)

    for n in nodes_in:
        nid = node_id(n["slice"], n["cluster"])

        raw_doc_weights = n.get("doc_weights", {}) or {}

        enriched_docs = {
            doc_id: {
                "weight": float(weight),
                "filepath": doc_map.get(doc_id)
            }
            for doc_id, weight in raw_doc_weights.items()
        }

        nodes[nid] = {
            "id": nid,
            "slice": n["slice"],
            "cluster": n["cluster"],
            "size": int(n.get("size", 1)),

            # UI-facing enrichment
            "docs": enriched_docs,

            # optional convenience metric for D3 sizing/filtering
            "doc_mass": float(sum(raw_doc_weights.values()))
        }


    # LINKS (pure structure)

    links = [
        {
            "source": l["source"] if isinstance(l["source"], str) else node_id(*l["source"]),
            "target": l["target"] if isinstance(l["target"], str) else node_id(*l["target"]),

            "similarity": float(l.get("similarity", 0.0)),
            "weight": float(l.get("score", l.get("similarity", 0.0))),

            "from_slice": l.get("from_slice"),
            "to_slice": l.get("to_slice"),
        }
        for l in links_in
    ]

    # deduplicate links
    seen = set()
    clean_links = []

    for l in links:
        key = (l["source"], l["target"])
        if key in seen:
            continue
        seen.add(key)
        clean_links.append(l)

    return {
        "token": token,
        "nodes": list(nodes.values()),
        "links": clean_links
    }



# PIPELINE


def build_all():
    data = load_graph()
    doc_map = load_doc_map()

    outputs = {}

    for token in CONCEPT_SETS.keys():
        if token not in data:
            continue

        logger.info(f"[tier3] processing token={token}")

        outputs[token] = build_d3_graph(token, data, doc_map)

    return outputs


def write(outputs: Dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"data": outputs}, f, indent=2)

    logger.info(f"[tier3] wrote {OUTPUT_PATH}")





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=False)
    args = ap.parse_args()

    data = load_graph()
    doc_map = load_doc_map()

    if args.token:
        write({
            args.token: build_d3_graph(args.token, data, doc_map)
        })
    else:
        write(build_all())


if __name__ == "__main__":
    main()