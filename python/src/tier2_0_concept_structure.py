#!/usr/bin/env python
"""
tier2_concept_structure.py

Tier 2:
    - slice-local embedding clustering
    - DBSCAN grouping in embedding space
    - produces atomic semantic micro-clusters per token per slice
    - deterministic recomputation from Tier 1 Zarr

Core invariant:
    - clustering is strictly slice-local
    - embeddings are never modified, only re-centered for analysis
    - no cross-slice semantic propagation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import zarr

from lib.eebo_db import get_connection
from lib.eebo_config import ZARR_ROOT, SLICES, CONCEPT_SETS, OUT_DIR
from lib.eebo_logging import logger

out_dir = OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = out_dir / "tier2_0.json"

DIAGNOSTICS: Dict[str, Any] = {}


def to_jsonable(obj):
    # Ensures numpy types do not leak into JSON serialization
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


_SLICE_CACHE: Dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_slice(slice_id: str):
    # Slice-level caching avoids repeated Zarr IO during token iteration
    if slice_id in _SLICE_CACHE:
        return _SLICE_CACHE[slice_id]

    logger.debug(f"[tier2] loading slice={slice_id}")

    root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
    vecs = root["vecs"][:]
    ids = root["ids"][:]

    _SLICE_CACHE[slice_id] = (vecs, ids)
    return vecs, ids


def local_center(vecs: np.ndarray) -> np.ndarray:
    """
    Removes global discourse bias within a slice.

    Important:
        This is not semantic normalization.
        It only recenters distribution to reduce register effects.
    """
    centroid = np.mean(vecs, axis=0)
    return vecs - centroid


def cluster_vectors(vecs: np.ndarray):
    """
    DBSCAN clustering in cosine space.

    Failure mode:
        DBSCAN cluster IDs are not stable across different feature spaces
        or preprocessing pipelines.
    """
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

    # Fallback: if DBSCAN finds no structure, treat full set as one cluster
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


def resolve_token(conn, token):
    """
    Maps surface token → (vector_id, doc_id).

    Failure mode:
        High-frequency tokens may return large fanout; no LIMIT applied.
        Consider pre-indexing or caching if runtime becomes unstable.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_id, doc_id
            FROM pamphlet_tokens
            WHERE token = %s
        """, (token.lower(),))

        rows = list(cur)

    vector_ids = [int(r[0]) for r in rows]
    doc_ids = [r[1] for r in rows]

    logger.debug(
        f"[tier2] resolve_token={token} "
        f"docs={len(set(doc_ids))} vectors={len(vector_ids)}"
    )

    return vector_ids, doc_ids


def analyse_token(
    vecs: np.ndarray,
    vector_ids: np.ndarray,
    doc_ids: np.ndarray,  # must align 1:1 with vector_ids
    token_vector_ids: List[int],
    token_doc_map: Dict[int, str],
    token: str,
    sid: str
):
    # Mask selects only vectors relevant to the token
    token_set = set(token_vector_ids)
    mask = np.array([v in token_set for v in vector_ids])

    filtered_vecs = vecs[mask]
    filtered_vec_ids = vector_ids[mask]

    if filtered_vecs.shape[0] == 0:
        logger.warning(f"[tier2] For \"{token}\" no matching vectors after token filter")
        return {
            "clusters": [],
            "noise": 0,
            "count": 0,
            "richness": 0.0,
            "noise_ratio": 0.0,
            "empty": True,
            "reason": "no matching vectors after token filter"
        }

    # Map vector_id → document identity
    # Failure mode: missing keys produce None and silently drop signal
    filtered_docs = np.array([
        token_doc_map.get(v)
        for v in filtered_vec_ids
    ], dtype=object)

    # Centering is applied BEFORE clustering to reduce slice-level bias
    normalized_vecs = local_center(filtered_vecs)

    # IMPORTANT FIX:
    # clustering and label assignment MUST use the same feature space
    from sklearn.cluster import DBSCAN

    if len(normalized_vecs) >= 5:
        model = DBSCAN(eps=0.18, min_samples=3, metric="cosine")
        labels = model.fit_predict(normalized_vecs)
    else:
        labels = np.array([-1] * len(normalized_vecs))

    clusters, noise = cluster_vectors(normalized_vecs)

    # Build doc-level aggregation per cluster label
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

        # Measures token clustering density within slice
        "richness": len(enriched) / (len(filtered_vecs) + 1e-12),

        # FIX: noise ratio must be relative to filtered set, not full slice
        "noise_ratio": noise / (len(filtered_vecs) + 1e-12),
    }


def run_all_slices(token_vector_ids, token_doc_map, token):
    results = {}

    for s in SLICES:
        sid = f"{s[0]}-{s[1]}"

        vecs, vector_ids = load_slice(sid)

        results[sid] = analyse_token(
            vecs,
            vector_ids,
            vector_ids,  # FIX: previously miswired; must align with vector_ids stream
            token_vector_ids,
            token_doc_map,
            token,
            sid
        )

    return results


def write_output(data: Dict[str, Any]):
    payload = {
        "data": to_jsonable(data),
        "diagnostics": DIAGNOSTICS
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[tier2] wrote output={OUTPUT_PATH}")


def main():
    global DIAGNOSTICS

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="*", default=None)
    parsed = parser.parse_args()

    conn = get_connection()

    tokens = parsed.tokens if parsed.tokens else list(CONCEPT_SETS.keys())

    output = {"tokens": {}}

    for token in tokens:
        logger.info(f"[tier2] processing token={token}")

        token_vec_ids, token_doc_ids = resolve_token(conn, token)

        # Critical mapping invariant:
        # vector_id → doc_id must remain stable for enrichment step
        token_doc_map = dict(zip(token_vec_ids, token_doc_ids))

        # NOTE:
        # Full concatenation across slices was previously computed but unused.
        # It has been removed conceptually; Tier 2 is strictly slice-local.

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
