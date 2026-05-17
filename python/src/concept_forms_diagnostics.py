#!/usr/bin/env python3
"""
concept_nearest_neighbours.py

Tier 1 concept validity + neighbourhood diagnostics over EEBO embeddings.

Core function:
- test whether curated orthographic forms are recovered in semantic space

Critical schema:
- Zarr stores:
    vecs: float32 [N, D]
    ids:  int64   [N]  (vector_id, NOT token)

- lexical forms live in Postgres, not Zarr

CONCEPT VALIDATION METRIC

hit_rate = |(NN_k ∩ effective_forms)| / |effective_forms|

where:
    effective_forms = forms - false_positives

## Results

| Type	        | Behaviour	                         | Examples
+---------------+------------------------------------+------------------------------------
| Anchors	    | high hit_rate, stable clustering	 | LAW, KING
| Institutions	| mid/high structured clusters	     | PARLIAMENT, COMMONWEALTH
| Fields	    | low hit_rate, diffuse embeddings	 | LIBERTY, DIVINE

## Expand false positives asymmetrically

- high-frequency near-misses
- morphologically similar but semantically distinct clusters
- historically relevant false friends (libertine-type drift is classic)

## Add “neutral neighbours sampling”

For each concept centroid, sample:

- top-k neighbours (already implemented)
- mid-distance band (distributional boundary zone)
- random baseline slice (corpus control distribution)

This allows separation of:
- semantic clustering signal
- corpus frequency bias
- embedding manifold density effects
"""

import numpy as np
import zarr
from pathlib import Path

from lib.eebo_config import CONCEPT_SETS, ZARR_ROOT
from lib.eebo_logging import logger
from lib.eebo_db import get_connection


# ----------------------------
# Zarr loading
# ----------------------------

def iter_slices():
    base = Path(ZARR_ROOT) / "tier1"
    for p in sorted(base.iterdir()):
        if p.is_dir():
            yield p


def load_slice(slice_path):
    root = zarr.open_group(str(slice_path), mode="r")

    if "vecs" not in root or "ids" not in root:
        raise KeyError(
            f"Invalid slice {slice_path}. Expected ['vecs','ids'], "
            f"got {list(root.array_keys())}"
        )

    return root["vecs"][:], root["ids"][:]


def build_global_space():
    vecs_all, ids_all = [], []

    for s in iter_slices():
        logger.info(f"[load] {s.name}")
        v, i = load_slice(s)
        vecs_all.append(v)
        ids_all.append(i)

    vecs = np.vstack(vecs_all).astype(np.float32)
    ids = np.concatenate(ids_all).astype(np.int64)

    # cosine normalisation invariant
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms

    return vecs, ids


# ----------------------------
# lexical resolution layer
# ----------------------------

def load_vector_id_to_token(conn):
    logger.info("[db] loading vector_id → token map")

    cur = conn.cursor()
    cur.execute("""
        SELECT vector_id, LOWER(token)
        FROM pamphlet_tokens
    """)

    return {vid: tok for vid, tok in cur}


# ----------------------------
# geometry utilities
# ----------------------------

def centroid(vectors):
    c = vectors.mean(axis=0)
    n = np.linalg.norm(c)
    return c / n if n > 0 else c


def topk(vecs, query, k=100):
    sims = vecs @ query

    if k >= len(sims):
        idx = np.argsort(-sims)
    else:
        idx = np.argpartition(-sims, k)[:k]
        idx = idx[np.argsort(-sims[idx])]

    return idx, sims[idx]


def mid_band(vecs, query, low=0.3, high=0.6, sample_n=200):
    sims = vecs @ query
    mask = (sims >= low) & (sims <= high)

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), np.array([])

    if len(idx) > sample_n:
        idx = np.random.choice(idx, sample_n, replace=False)

    return idx, sims[idx]


def random_baseline(vecs, sample_n=200):
    idx = np.random.choice(len(vecs), sample_n, replace=False)
    sims = np.zeros(sample_n)
    return idx, sims


# ----------------------------
# concept analysis
# ----------------------------

def analyse_concept(name, concept, vecs, ids, id2tok, k=100):

    forms = {f.lower() for f in concept["forms"]}
    false_positives = {f.lower() for f in concept.get("false_positives", set())}

    effective_forms = forms - false_positives

    logger.info(f"[concept] {name}")

    if not effective_forms:
        logger.warning(f"[{name}] no effective forms after filtering")
        return None

    # resolve vector_ids for forms
    form_vector_ids = {
        vid for vid, tok in id2tok.items()
        if tok in effective_forms
    }

    if not form_vector_ids:
        logger.warning(f"[{name}] no vector_ids found for forms")
        return None

    mask = np.isin(ids, list(form_vector_ids))
    form_vecs = vecs[mask]

    if len(form_vecs) == 0:
        logger.warning(f"[{name}] no vectors found in embedding space")
        return None

    c = centroid(form_vecs)

    # ----------------------------
    # neighbourhood sampling
    # ----------------------------

    nn_idx, nn_sims = topk(vecs, c, k=k)
    mid_idx, _ = mid_band(vecs, c)
    rand_idx, _ = random_baseline(vecs)

    # token resolution
    nn_tokens = [id2tok.get(i, "<UNK>") for i in ids[nn_idx]]

    hits = [t for t in nn_tokens if t in effective_forms]

    hit_rate = len(hits) / len(effective_forms)

    result = {
        "concept": name,
        "forms": len(forms),
        "effective_forms": len(effective_forms),
        "found_vectors": int(len(form_vecs)),
        "hit_rate": hit_rate,
        "hits": hits,
        "top_k": nn_tokens[:20],
        "mid_band_size": len(mid_idx),
        "random_baseline_size": len(rand_idx),
    }

    logger.info(
        f"[{name}] eff_forms={len(effective_forms)} "
        f"found={len(form_vecs)} "
        f"hit_rate={hit_rate:.3f}"
    )

    return result


# ----------------------------
# main
# ----------------------------

def main():
    conn = get_connection()

    logger.info("Building global embedding space")
    vecs, ids = build_global_space()

    logger.info(f"Global vectors: {len(vecs):,}")

    id2tok = load_vector_id_to_token(conn)

    results = []

    for name, concept in CONCEPT_SETS.items():
        res = analyse_concept(name, concept, vecs, ids, id2tok, k=100)
        if res:
            results.append(res)

    out_path = Path(ZARR_ROOT) / "concept_nearest_neighbours.json"

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote {out_path}")

    conn.close()


if __name__ == "__main__":
    main()
