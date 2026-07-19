#!/usr/bin/env python
"""
tier2_5_to_d3_json.py

Converts Tier 2.5 semantic neighbourhood output into a
frontend-ready JSON structure for D3 / SolidJS exploration.

DESIGN GOAL

This script does NOT analyse data.

It restructures it into an interactive semantic graph:

    concept
        ↳ slice
            ↳ instances
                ↳ neighbours
                    ↳ doc_id (XML resolution target)

This enables:
    - slice navigation (temporal drift browsing)
    - instance inspection (semantic event view)
    - neighbour exploration (field structure view)
    - external XML linking via doc_id

CRITICAL INVARIANT

We preserve instance identity exactly as produced by Tier 2.5:
    - no aggregation
    - no centroiding
    - no recomputation of embeddings

This is a pure serialization / transformation layer.



"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from lib.eebo_config import OUT_DIR, XML_ROOT_DIR
from lib.eebo_logging import logger
from tier2_5_concept_neighbours_temporal import OUTPUT_PATH as INPUT_PATH

OUTPUT_PATH = OUT_DIR / "tier2_5_d3.json"


def load():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_filepath(fp: str) -> str:
    if not fp:
        return None

    p = Path(fp).expanduser().resolve()

    try:
        p = p.relative_to(XML_ROOT_DIR)
    except ValueError:
        # fallback: keep absolute but normalized
        return str(p)

    return p.as_posix()


# Transform to UI-ready structure
def transform(data):
    out = {
        "k": data.get("k"),
        "concepts": {},
        "index": {}
    }

    for concept, payload in data["concepts"].items():
        if payload.get("empty"):
            continue

        concept_node = {
            "name": concept,
            "forms": payload.get("forms", []),
            "n_instances": payload.get("n_instances", 0),
            "slices": {}
        }

        slices = defaultdict(list)

        for inst in payload["instances"]:
            slice_id = inst.get("slice", "unknown")

            node = {
                "vector_id": inst["vector_id"],
                "token": inst.get("token"),
                "doc_id": inst.get("doc_id"),
                "filepath": normalize_filepath(inst.get("filepath")),
                "slice": slice_id,
                "xy": inst.get("xy"),
                "neighbours": inst.get("neighbours", [])
            }

            slices[slice_id].append(node)

            out["index"][inst["vector_id"]] = {
                "concept": concept,
                "slice": slice_id,
                "token": inst.get("token"),
                "doc_id": inst.get("doc_id"),
                "filepath": normalize_filepath(inst.get("filepath")),
                "xy": inst.get("xy")
            }

        # assemble slice structure
        for sid, instances in slices.items():
            concept_node["slices"][sid] = {
                "slice_id": sid,
                "instances": instances,
                "n_instances": len(instances)
            }

        out["concepts"][concept] = concept_node

    return out


# --------------------------------------
# Main
# --------------------------------------
def main():
    data = load()
    transformed = transform(data)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2)

    print(f"[tier2.5 to d3] wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
