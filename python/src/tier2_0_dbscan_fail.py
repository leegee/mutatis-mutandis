#!/usr/bin/env python
"""
tier2_0_dbscan_fail.py

Tier 2:
    - slice-local embedding clustering
    - DBSCAN grouping in embedding space
    - produces atomic semantic micro-clusters per token per slice
    - deterministic recomputation from Tier 1 Zarr
    - No links between nodes/clusters
    - Plot global bias.

Core invariants:
    - clustering is strictly slice-local
    - embeddings are never modified in stored outputs
    - raw semantic geometry is preserved
    - document provenance is preserved at all stages

IMPORTANT:

    We now preserve BOTH:

        1. raw semantic geometry
        2. centered analytic geometry

    because these represent different things.

RAW SPACE:
    Historical/discourse semantic field.

CENTERED SPACE:
    Local slice-relative deviation from discourse baseline.

Tier 2.7 should generally operate on RAW geometry unless
explicitly experimenting with relative semantic structure.
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


# ------------------------------------------------------------------
# JSON safety
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Slice cache
# ------------------------------------------------------------------

_SLICE_CACHE: Dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_slice(slice_id: str):
    """
    Loads full slice embeddings + vector ids.

    Invariant:
        ids[i] aligns 1:1 with vecs[i]
    """

    if slice_id in _SLICE_CACHE:
        return _SLICE_CACHE[slice_id]

    logger.info(f"[tier2] loading_slice={slice_id}")

    root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")

    vecs = root["vecs"][:]
    ids = root["ids"][:]

    _SLICE_CACHE[slice_id] = (vecs, ids)

    logger.info(
        f"[tier2] loaded_slice={slice_id} "
        f"vecs={len(vecs)}"
    )

    return vecs, ids


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

def local_center(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Removes slice-global embedding field.

    IMPORTANT:
        This is NOT semantic normalization.
        It is an analytic projection only.

    Returns:
        centered_vecs,
        global_centroid
    """

    centroid = np.mean(vecs, axis=0)

    centered = vecs - centroid

    return centered, centroid


def summarise_clusters(
    vecs: np.ndarray,
    labels: np.ndarray,
    vector_ids: np.ndarray,
    doc_ids: np.ndarray
):
    """
    Produces cluster summaries while preserving provenance.

    IMPORTANT:
        Cluster IDs are local-only DBSCAN labels.
        They are NOT stable semantic identities.
    """

    clusters = defaultdict(list)
    cluster_members = defaultdict(list)

    noise = 0

    for i, lab in enumerate(labels):

        if lab == -1:
            noise += 1
            continue

        cid = int(lab)

        clusters[cid].append(vecs[i])

        cluster_members[cid].append({
            "vector_id": int(vector_ids[i]),
            "doc_id": str(doc_ids[i])
        })

    summary = []

    for cid, members in clusters.items():

        members_arr = np.stack(members)

        centroid = np.mean(members_arr, axis=0)
        centroid = centroid / (
            np.linalg.norm(centroid) + 1e-12
        )

        raw_members = cluster_members[cid]

        doc_weights = defaultdict(int)

        for m in raw_members:
            doc_weights[m["doc_id"]] += 1

        summary.append({
            "cluster_id": cid,

            "size": int(len(members_arr)),

            "centroid": centroid.astype(np.float32).tolist(),

            "variance": float(
                np.var(members_arr, axis=0).mean()
            ),

            "weight": float(
                len(members_arr) / (len(vecs) + 1e-12)
            ),

            # provenance
            "members": raw_members,

            "doc_ids": sorted(doc_weights.keys()),

            "doc_weights": dict(doc_weights),

            "doc_mass": int(sum(doc_weights.values()))
        })

    return summary, int(noise)


# ------------------------------------------------------------------
# Token resolution
# ------------------------------------------------------------------

def resolve_token(conn, token):
    """
    Maps token -> vector ids + doc ids.

    IMPORTANT:
        token table is already lowercased on insert.
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

    logger.info(
        f"[tier2] resolve_token={token} "
        f"vectors={len(vector_ids)} "
        f"docs={len(set(doc_ids))}"
    )

    return vector_ids, doc_ids


# ------------------------------------------------------------------
# Token analysis
# ------------------------------------------------------------------

def analyse_token(
    vecs: np.ndarray,
    vector_ids: np.ndarray,
    token_vector_ids: List[int],
    token_doc_map: Dict[int, str],
    token: str,
    sid: str
):

    token_set = set(token_vector_ids)

    mask = np.array(
        [v in token_set for v in vector_ids]
    )

    filtered_vecs = vecs[mask]
    filtered_vec_ids = vector_ids[mask]

    if filtered_vecs.shape[0] == 0:

        logger.warning(
            f"[tier2] token={token} "
            f"slice={sid} "
            f"empty_after_filter"
        )

        return {
            "clusters": {
                "raw": [],
                "centered": []
            },
            "noise": {},
            "count": 0,
            "empty": True
        }

    filtered_docs = np.array([
        token_doc_map.get(v)
        for v in filtered_vec_ids
    ], dtype=object)

    logger.info(
        f"[tier2] token={token} "
        f"slice={sid} "
        f"occurrences={len(filtered_vecs)} "
        f"docs={len(set(filtered_docs.tolist()))}"
    )

    # --------------------------------------------------------------
    # Global slice field
    # --------------------------------------------------------------

    centered_vecs, global_centroid = local_center(filtered_vecs)

    logger.info(
        f"[tier2] token={token} "
        f"slice={sid} "
        f"global_centroid_norm="
        f"{float(np.linalg.norm(global_centroid)):.6f}"
    )

    # --------------------------------------------------------------
    # RAW clustering
    # --------------------------------------------------------------

    from sklearn.cluster import DBSCAN

    if len(filtered_vecs) >= 5:

        raw_model = DBSCAN(
            eps=0.18,
            min_samples=3,
            metric="cosine"
        )

        raw_labels = raw_model.fit_predict(
            filtered_vecs
        )

    else:
        raw_labels = np.array(
            [-1] * len(filtered_vecs)
        )

    raw_clusters, raw_noise = summarise_clusters(
        filtered_vecs,
        raw_labels,
        filtered_vec_ids,
        filtered_docs
    )

    logger.info(
        f"[tier2] token={token} "
        f"slice={sid} "
        f"raw_clusters={len(raw_clusters)} "
        f"raw_noise={raw_noise}"
    )

    # --------------------------------------------------------------
    # CENTERED clustering
    # --------------------------------------------------------------

    if len(centered_vecs) >= 5:

        centered_model = DBSCAN(
            eps=0.18,
            min_samples=3,
            metric="cosine"
        )

        centered_labels = centered_model.fit_predict(
            centered_vecs
        )

    else:
        centered_labels = np.array(
            [-1] * len(centered_vecs)
        )

    centered_clusters, centered_noise = summarise_clusters(
        centered_vecs,
        centered_labels,
        filtered_vec_ids,
        filtered_docs
    )

    logger.info(
        f"[tier2] token={token} "
        f"slice={sid} "
        f"centered_clusters={len(centered_clusters)} "
        f"centered_noise={centered_noise}"
    )

    return {

        "global_centroid":
            global_centroid.astype(np.float32).tolist(),

        "count":
            int(len(filtered_vecs)),

        "doc_count":
            int(len(set(filtered_docs.tolist()))),

        "clusters": {

            "raw":
                raw_clusters,

            "centered":
                centered_clusters
        },

        "noise": {

            "raw":
                int(raw_noise),

            "centered":
                int(centered_noise)
        }
    }


# ------------------------------------------------------------------
# Slice driver
# ------------------------------------------------------------------

def run_all_slices(
    token_vector_ids,
    token_doc_map,
    token
):

    results = {}

    for s in SLICES:

        sid = f"{s[0]}-{s[1]}"

        vecs, vector_ids = load_slice(sid)

        results[sid] = analyse_token(
            vecs,
            vector_ids,
            token_vector_ids,
            token_doc_map,
            token,
            sid
        )

    return results


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def write_output(data: Dict[str, Any]):

    payload = {
        "data": to_jsonable(data),
        "diagnostics": DIAGNOSTICS
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"[tier2] wrote_output={OUTPUT_PATH}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    global DIAGNOSTICS

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tokens",
        nargs="*",
        default=None
    )

    parsed = parser.parse_args()

    conn = get_connection()

    tokens = (
        parsed.tokens
        if parsed.tokens
        else list(CONCEPT_SETS.keys())
    )

    output = {"tokens": {}}

    for token in tokens:

        logger.info(
            f"[tier2] processing_token={token}"
        )

        token_vec_ids, token_doc_ids = resolve_token(
            conn,
            token
        )

        token_doc_map = dict(
            zip(token_vec_ids, token_doc_ids)
        )

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

