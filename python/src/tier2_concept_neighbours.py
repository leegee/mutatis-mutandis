#!/usr/bin/env python
"""
tier2_concept_neighbours.py

Instance-level semantic neighbourhood extraction for EEBO concepts.

Rationale:

Early Modern English lexical forms are sparse and unstable.
Aggregating to centroids risks collapsing meaningful rhetorical
and ideological variation.

This pipeline therefore operates at the level of individual
occurrences, treating each embedding as a distinct historical
semantic event.

For each occurrence:
    - retrieve nearest semantic neighbours in embedding space
    - preserve token + document provenance
    - analyse distribution of neighbourhoods across a concept

This supports later analysis of:
    - semantic drift as distributional movement
    - rhetorical clustering of usage events
    - conceptual field fragmentation and reorganisation
"""

from __future__ import annotations

import json
from typing import Dict, Any
import numpy as np
import zarr

from lib.eebo_config import CONCEPT_SETS, ZARR_ROOT, OUT_DIR
from lib.eebo_db import get_connection
from lib.eebo_logging import logger


OUTPUT_PATH = OUT_DIR / "concept_neighbours.json"

K_NEIGHBOURS = 25


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

    # FAST COSINE PREP:
    # normalise once so cosine becomes dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    vecs = vecs / norms

    return vecs, ids


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


# FAST nearest neighbours (cosine = dot product after normalisation)
def nearest(vecs, ids, vec_to_token, query_vec, k):
    sims = vecs @ query_vec  # already normalised

    # partial top-k (fast selection)
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


def process_concept(vecs, ids, vec_to_token, vec_to_doc, concept_name, concept):
    forms = {f.lower() for f in concept["forms"]}
    logger.info(f"[concept_neigh]      processing_form={concept_name} forms={len(forms)}")

    mask = np.array([
        vec_to_token.get(int(v)) in forms
        for v in ids
    ])

    concept_vecs = vecs[mask]
    concept_ids = ids[mask]

    if len(concept_vecs) == 0:
        logger.warning(f"[concept_neigh] empty_concept={concept_name}")
        return {
            "concept": concept_name,
            "forms": list(forms),
            "empty": True
        }

    results = []

    for i, vid in enumerate(concept_ids):
        if i % 50 == 0:
            logger.info(
                f"[concept_neigh] {concept_name} instance={i}/{len(concept_ids)}"
            )

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
            "neighbours": neighbours
        })

    return {
        "concept": concept_name,
        "forms": list(forms),
        "n_instances": int(len(concept_vecs)),
        "instances": results
    }



def main():
    logger.info("[concept_neigh] loading embeddings")
    vecs, ids = load_embeddings()

    logger.info("[concept_neigh] loading token index")
    vec_to_token, vec_to_doc = load_token_index()

    output: Dict[str, Any] = {
        "k": K_NEIGHBOURS,
        "concepts": {}
    }

    for concept_name, concept in CONCEPT_SETS.items():

        logger.info(f"[concept_neigh] START concept={concept_name}")

        output["concepts"][concept_name] = process_concept(
            vecs,
            ids,
            vec_to_token,
            vec_to_doc,
            concept_name,
            concept
        )

        logger.info(f"[concept_neigh] DONE concept={concept_name}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"[concept_neigh] wrote={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
