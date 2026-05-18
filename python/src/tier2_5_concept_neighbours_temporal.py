#!/usr/bin/env python
"""
tier2_5_concept_neighbours_temporal.py

Temporal extension of Tier 2 instance-level semantic neighbourhood extraction.

RATIONALE

Tier 2 demonstrated that Early Modern semantic structure is best captured
at the level of individual usage events rather than aggregated centroids.

However, Tier 2 collapses all occurrences across time, losing the ability to
observe conceptual drift.

Tier 2.5 restores temporal structure by re-introducing slice provenance
derived from Zarr storage layout.

KEY DESIGN PRINCIPLE

Slice identity is NOT inferred from tokens or embeddings.
It is recovered from the physical embedding partitioning:

    ZARR_ROOT/tier1/{slice_id}/

Each vector belongs to exactly one slice by construction.

This preserves:
    - instance-level semantic fidelity (Tier 2 invariant)
    - temporal ordering (new Tier 2.5 invariant)
    - no modification of embedding geometry

RESULT

Enables:
    - slice-level semantic field comparison
    - drift analysis of neighbour distributions
    - conceptual stability vs volatility measurement
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import zarr

from lib.eebo_config import CONCEPT_SETS, ZARR_ROOT, OUT_DIR
from lib.eebo_db import get_connection
from lib.eebo_logging import logger


OUTPUT_PATH = OUT_DIR / "tier2_5_concept_neighbours_temporal.json"
K_NEIGHBOURS = 25


# Slice reconstruction from Zarr layout
def build_vector_slice_map() -> Dict[int, str]:
    """
    Reconstruct vector_id → slice_id mapping from Zarr partitioning.

    Invariant:
        each vector_id appears in exactly one slice folder
    """

    mapping: Dict[int, str] = {}

    root = ZARR_ROOT / "tier1"

    for slice_dir in sorted(root.iterdir()):
        if not slice_dir.is_dir():
            continue

        slice_id = slice_dir.name
        z = zarr.open(slice_dir, mode="r")

        ids = z["ids"][:]

        for vid in ids:
            mapping[int(vid)] = slice_id

    logger.info(f"[tier2.5] slice_map_size={len(mapping)}")

    return mapping


# Embeddings (unchanged from Tier 2)
def load_embeddings():
    vecs_all = []
    ids_all = []

    root = ZARR_ROOT / "tier1"

    for slice_dir in sorted(root.iterdir()):
        if not slice_dir.is_dir():
            continue

        z = zarr.open(slice_dir, mode="r")

        vecs_all.append(z["vecs"][:])
        ids_all.append(z["ids"][:])

    vecs = np.concatenate(vecs_all, axis=0)
    ids = np.concatenate(ids_all, axis=0)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    return vecs, ids


# Token index
def load_token_index():
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_id, token, doc_id
            FROM pamphlet_tokens
        """)
        rows = list(cur)

    conn.close()

    vec_to_token = {}
    vec_to_doc = {}

    for vid, tok, doc in rows:
        vid = int(vid)
        vec_to_token[vid] = str(tok)
        vec_to_doc[vid] = doc

    return vec_to_token, vec_to_doc


# Fast cosine (dot product on normalised vectors)
def nearest(vecs, ids, vec_to_token, query_vec, k):
    sims = vecs @ query_vec
    sims = np.clip(sims, -1.0, 1.0)

    idx = np.argpartition(-sims, kth=min(k, len(sims) - 1))[:k]
    idx = idx[np.argsort(-sims[idx])]

    return [
        {
            "vector_id": int(ids[i]),
            "token": vec_to_token.get(int(ids[i])),
            "similarity": float(sims[i]),
        }
        for i in idx
    ]


# Concept processing (now slice-aware)
def process_concept(
    vecs,
    ids,
    vec_to_token,
    vec_to_doc,
    vec_to_slice,
    concept_name,
    concept
):

    forms = {f.lower() for f in concept["forms"]}
    logger.info(f"[tier2.5] processing={concept_name} forms={len(forms)}")

    mask = np.array([
        vec_to_token.get(int(v)) in forms
        for v in ids
    ])

    concept_vecs = vecs[mask]
    concept_ids = ids[mask]

    if len(concept_vecs) == 0:
        logger.warning(f"[tier2.5] empty={concept_name}")
        return {
            "concept": concept_name,
            "forms": list(forms),
            "empty": True
        }

    results = []

    for i, vid in enumerate(concept_ids):
        if i % 50 == 0:
            logger.info(f"[tier2.5] {concept_name} instance={i}/{len(concept_ids)}")

        vec = concept_vecs[i]

        neighbours = nearest(
            vecs,
            ids,
            vec_to_token,
            vec,
            K_NEIGHBOURS
        )

        results.append({
            "vector_id": int(vid),
            "token": vec_to_token.get(int(vid)),
            "doc_id": vec_to_doc.get(int(vid)),
            "slice": vec_to_slice.get(int(vid)),   # NEW
            "neighbours": neighbours
        })

    return {
        "concept": concept_name,
        "forms": list(forms),
        "n_instances": int(len(concept_vecs)),
        "instances": results
    }



def main():
    logger.info("[tier2.5] loading embeddings")
    vecs, ids = load_embeddings()

    logger.info("[tier2.5] loading token index")
    vec_to_token, vec_to_doc = load_token_index()

    logger.info("[tier2.5] building slice map")
    vec_to_slice = build_vector_slice_map()

    output: Dict[str, Any] = {
        "k": K_NEIGHBOURS,
        "concepts": {}
    }

    for concept_name, concept in CONCEPT_SETS.items():
        logger.info(f"[tier2.5] START {concept_name}")
        output["concepts"][concept_name] = process_concept(
            vecs,
            ids,
            vec_to_token,
            vec_to_doc,
            vec_to_slice,
            concept_name,
            concept
        )

        logger.info(f"[tier2.5] DONE {concept_name}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[tier2.5] wrote={OUTPUT_PATH}")


if __name__ == "__main__":
    main()