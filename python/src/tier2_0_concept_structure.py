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


_SLICE_CACHE: Dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_slice(slice_id: str):
    if slice_id in _SLICE_CACHE:
        return _SLICE_CACHE[slice_id]

    logger.debug(f"[tier2] loading slice={slice_id}")

    root = zarr.open(ZARR_ROOT / "tier1" / slice_id, mode="r")
    vecs = root["vecs"][:]
    ids = root["ids"][:]

    logger.debug(
        f"[tier2] slice_loaded={slice_id} vecs={vecs.shape[0]} ids={ids.shape[0]}"
    )

    _SLICE_CACHE[slice_id] = (vecs, ids)
    return vecs, ids


def cluster_vectors(vecs: np.ndarray):
    from sklearn.cluster import DBSCAN

    if len(vecs) < 5:
        return [], len(vecs)

    # --- PRIMARY CLUSTERING ---
    model = DBSCAN(
        eps=0.18,          # slightly more permissive (critical for EEBO noise)
        min_samples=3,     # allow micro-clusters
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

            # IMPORTANT: preserve real density signal
            "size": int(len(members)),

            "centroid": centroid.astype(np.float32).tolist(),

            # turbulence signal (used later in Tier 2.75)
            "variance": float(np.var(members, axis=0).mean()),

            # NEW: structural weight (important for branching)
            "weight": float(len(members) / (len(vecs) + 1e-12))
        })

    # --- FALLBACK: if DBSCAN collapses everything ---
    # THIS is critical for LIBERTY-like sparse slices
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


def resolve_token(conn, token: str) -> List[int]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT vector_id
            FROM pamphlet_tokens
            WHERE token = %s
        """, (token.lower(),))
        ids = [int(r[0]) for r in cur]

    logger.debug(f"[tier2] resolve_token={token} ids={len(ids)}")
    return ids


# identity diagnostics
def probe_identity(vec_ids: np.ndarray, target_ids: List[int]) -> Dict[str, Any]:
    vec_set = set(map(int, vec_ids[:200000]))
    tgt_set = set(map(int, target_ids))

    inter = vec_set & tgt_set

    missing = list(tgt_set - vec_set)
    extra = list(vec_set - tgt_set)

    return {
        "intersection_size": len(inter),
        "target_size": len(tgt_set),
        "vector_size_sample": len(vec_set),
        "coverage": float(len(inter) / (len(tgt_set) + 1e-12)),
        "missing_sample": missing[:10],
        "extra_sample": extra[:10],
    }


def analyse_token(
    vecs: np.ndarray,
    ids: np.ndarray,
    target_ids: List[int],
    token: str,
    sid: str
):
    mask = np.isin(ids, target_ids)
    filtered = vecs[mask]

    diag = probe_identity(ids, target_ids)

    logger.debug(
        f"[tier2] token={token} slice={sid} "
        f"match={filtered.shape[0]} total={vecs.shape[0]} "
        f"coverage={diag['coverage']:.4f}"
    )

    clusters, noise = cluster_vectors(filtered)

    # preserve intra-slice richness signal
    richness = len(clusters) / (len(filtered) + 1e-12)

    return {
        "clusters": clusters,
        "noise": int(noise),
        "count": int(len(filtered)),
        "richness": float(richness),
        "noise_ratio": noise / (len(vecs) + 1e-12), # Avoid div by zero
        "diagnostics": diag
    }


def run_all_slices(token_ids: List[int], token: str) -> Dict[str, Any]:
    results = {}

    for s in SLICES:
        sid = f"{s[0]}-{s[1]}"

        vecs, ids = load_slice(sid)

        results[sid] = analyse_token(vecs, ids, token_ids, token, sid)

    return results


def write_output(data: Dict[str, Any]) -> None:
    global DIAGNOSTICS

    payload = {
        "data": to_jsonable(data),
        "diagnostics": DIAGNOSTICS
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"[tier2] wrote output={OUTPUT_PATH}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--tokens",
        nargs="*",
        default=None,
        help="Tokens to analyse (default: CONCEPT_SETS keys)"
    )

    return p.parse_args()



def main():
    global DIAGNOSTICS

    args = parse_args()

    conn = get_connection()

    tokens = args.tokens if args.tokens else list(CONCEPT_SETS.keys())

    logger.info(f"[tier2] start tokens={len(tokens)} mode={'CLI' if args.tokens else 'CONCEPT_SETS'}")

    output = {"tokens": {}}

    for token in tokens:
        logger.info(f"[tier2] processing token={token}")

        token_ids = resolve_token(conn, token)

        # global identity probe
        all_vecs = []
        all_ids = []

        for s in SLICES:
            sid = f"{s[0]}-{s[1]}"
            v, i = load_slice(sid)
            all_vecs.append(v)
            all_ids.append(i)

        all_vecs = np.concatenate(all_vecs)
        all_ids = np.concatenate(all_ids)

        DIAGNOSTICS[token] = probe_identity(all_ids, token_ids)

        logger.info(
            f"[tier2] token={token} GLOBAL coverage={DIAGNOSTICS[token]['coverage']:.6f}"
        )

        output["tokens"][token] = run_all_slices(token_ids, token)

    conn.close()

    write_output(output)

    logger.info("[tier2] complete")


if __name__ == "__main__":
    main()
