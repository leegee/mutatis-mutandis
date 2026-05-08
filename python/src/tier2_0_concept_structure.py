#!/usr/bin/env python
"""
tier2_concept_structure.py

Tier 2:
    - slice-local embedding clustering
    - DBSCAN grouping
    - produces atomic semantic "micro-clusters"
    - persistent JSON artefact output

Invariant:
    - no cross-slice clustering
    - no embedding modification
    - deterministic recomputation from Tier 1 Zarr

Key upgrade:
    - correct vector_id → doc_id propagation
    - slice-safe document projection
    - cluster-level document support weighting
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import zarr

from lib.eebo_db import get_connection
from lib.eebo_config import ZARR_ROOT, SLICES, CONCEPT_SETS
from lib.eebo_logging import logger

out_dir = ZARR_ROOT / "tier2"
out_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = out_dir / "tier2.json"

DIAGNOSTICS: Dict[str, Any] = {}


# ----------------------------
# JSON safety
# ----------------------------

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    return obj


# ----------------------------
# slice cache
# ----------------------------

_SLICE_CACHE: Dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_slice(slice_id: str):
    if slice_id in _SLICE_CACHE:
        return _SLICE_CACHE[slice_id]

    logger.debug(f"[tier2] loading slice={slice_id}")

    root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
    vecs = root["vecs"][:]
    ids = root["ids"][:]

    _SLICE_CACHE[slice_id] = (vecs, ids)
    return vecs, ids


# ----------------------------
# clustering
# ----------------------------

def cluster_vectors(vecs: np.ndarray):
    from sklearn.cluster import DBSCAN

    if len(vecs) < 5:
        return [], len(vecs)

    model = DBSCAN(
        eps=0.18,
        min_samples=3,
        metric="cosine"
    )

    labels = model.fit_predict(vecs)

    clusters = defaultdict(list)
    noise = 0

    for i, lab in enumerate(labels):
        if lab == -1:
            noise += 1
        else:
            clusters[int(lab)].append(vecs[i])

    summary = []

    for cid, members in clusters.items():
        members = np.stack(members)

        centroid = np.mean(members, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        summary.append({
            "cluster_id": int(cid),
            "size": int(len(members)),
            "centroid": centroid.astype(np.float32).tolist(),
            "variance": float(np.var(members, axis=0).mean()),
            "weight": float(len(members) / (len(vecs) + 1e-12))
        })

    if len(summary) == 0:
        centroid = np.mean(vecs, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        summary.append({
            "cluster_id": 0,
            "size": int(len(vecs)),
            "centroid": centroid.astype(np.float32).tolist(),
            "variance": float(np.var(vecs, axis=0).mean()),
            "weight": 1.0
        })

    return summary, int(noise)


# ----------------------------
# DB resolution
# ----------------------------

def resolve_token(conn, token):
    """
    Returns:
        vector_id → doc_id mapping (critical invariant)
    """

    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_id, doc_id
            FROM pamphlet_tokens
            WHERE token = %s
        """, (token.lower(),))

        rows = list(cur)

    vector_ids = [int(r[0]) for r in rows]
    doc_ids = [r[1] for r in rows]  # STRING IDS

    logger.debug(
        f"[tier2] resolve_token={token} "
        f"docs={len(set(doc_ids))} vectors={len(vector_ids)}"
    )

    return vector_ids, doc_ids


# ----------------------------
# core analysis
# ----------------------------

def analyse_token(
    vecs: np.ndarray,
    vector_ids: np.ndarray,
    doc_ids: np.ndarray,
    token_vector_ids: List[int],
    token_doc_map: Dict[int, str],
    token: str,
    sid: str
):

    mask = np.isin(vector_ids, token_vector_ids)

    filtered_vecs = vecs[mask]
    filtered_vec_ids = vector_ids[mask]

    # CRITICAL FIX: correct doc projection
    filtered_docs = np.array([
        token_doc_map.get(v)
        for v in filtered_vec_ids
    ], dtype=object)

    clusters, noise = cluster_vectors(filtered_vecs)

    # DBSCAN re-alignment for doc assignment
    from sklearn.cluster import DBSCAN

    if len(filtered_vecs) >= 5:
        model = DBSCAN(eps=0.18, min_samples=3, metric="cosine")
        labels = model.fit_predict(filtered_vecs)
    else:
        labels = np.array([-1] * len(filtered_vecs))

    doc_map = defaultdict(lambda: defaultdict(int))

    for i, lab in enumerate(labels):
        if lab == -1:
            continue

        doc = filtered_docs[i]
        if doc is None:
            continue

        doc_map[int(lab)][doc] += 1

    enriched = []

    for c in clusters:
        cid = c["cluster_id"]

        doc_weights = dict(doc_map[cid])

        enriched.append({
            **c,
            "doc_ids": list(doc_weights.keys()),
            "doc_weights": doc_weights,
            "doc_mass": sum(doc_weights.values())
        })

    return {
        "clusters": enriched,
        "noise": int(noise),
        "count": int(len(filtered_vecs)),
        "richness": len(enriched) / (len(filtered_vecs) + 1e-12),
        "noise_ratio": noise / (len(vecs) + 1e-12),
    }


# ----------------------------
# pipeline
# ----------------------------

def run_all_slices(token_vector_ids, token_doc_map, token):

    results = {}

    for s in SLICES:
        sid = f"{s[0]}-{s[1]}"

        vecs, vector_ids = load_slice(sid)

        results[sid] = analyse_token(
            vecs,
            vector_ids,
            vector_ids,
            token_vector_ids,
            token_doc_map,
            token,
            sid
        )

    return results


# ----------------------------
# output
# ----------------------------

def write_output(data: Dict[str, Any]):

    payload = {
        "data": to_jsonable(data),
        "diagnostics": DIAGNOSTICS
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[tier2] wrote output={OUTPUT_PATH}")


# ----------------------------
# main
# ----------------------------

def main():

    global DIAGNOSTICS

    args = argparse.ArgumentParser()
    args.add_argument("--tokens", nargs="*", default=None)
    parsed = args.parse_args()

    conn = get_connection()

    tokens = parsed.tokens if parsed.tokens else list(CONCEPT_SETS.keys())

    output = {"tokens": {}}

    for token in tokens:

        logger.info(f"[tier2] processing token={token}")

        token_vec_ids, token_doc_ids = resolve_token(conn, token)

        # CRITICAL FIX: vector → doc mapping
        token_doc_map = dict(zip(token_vec_ids, token_doc_ids))

        all_vecs = []
        all_ids = []

        for s in SLICES:
            vecs, ids = load_slice(f"{s[0]}-{s[1]}")
            all_vecs.append(vecs)
            all_ids.append(ids)

        all_vecs = np.concatenate(all_vecs)
        all_ids = np.concatenate(all_ids)

        output["tokens"][token] = run_all_slices(
            token_vec_ids,
            token_doc_map,
            token
        )

    conn.close()

    write_output(output)

    logger.info("[tier2] complete")


if __name__ == "__main__":
    main()
