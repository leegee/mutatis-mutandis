#!/usr/bin/env python

"""
Compute semantic drift across time slices using optimal transport over
occurrence-level neighbourhood distributions.

Pipeline:

1. For each token and time slice:
   - Retrieve token_occurrence_ids from DB
   - Map to vector positions via id-aligned vector store
   - Query FAISS for local neighbourhoods per occurrence

2. Build a weighted semantic cloud:
   - X_accum: neighbour vectors
   - w_accum: kernel-weighted similarities

3. Compress cloud into a small measure:
   - k-means-like clustering (OT_CLUSTERS)
   - produces centroids + weights

4. Compute drift:
   - Wasserstein distance between consecutive slice measures

5. Output per-slice:
   - drift scalar
   - top neighbors (aggregated mass)
   - top documents
   - compressed geometry (centroids + weights) for visualization

Critical invariants:
- FAISS IDs must equal token_occurrence_id
- vectors.npz must be id-aligned with FAISS index
- DB token_occurrence_id must match embedding pipeline

Failure modes:
- Missing slice data = empty_slice emitted
- ID mismatch = neighbors silently dropped
- No valid neighbors = zero drift
"""

import json
import argparse
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass

import ot
from sklearn.metrics import pairwise_distances

from mb_embedding_pipeline import load_id_vectors
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR, SLICES, CONCEPT_SETS
from lib.wordlist import STOPWORDS


OUT_PATH = OUT_DIR / "drift_state.json"

K_NEIGHBORS = 5
TOP_K_NEIGHBORS = 15
OT_CLUSTERS = 12   # controls geometry resolution

_FAISS_CACHE = {}
_VEC_CACHE = {}

USE_KERNEL_DEFAULT = True
KERNEL_ALPHA_DEFAULT = 6.0


@dataclass
class OTMeasure:
    X: np.ndarray
    w: np.ndarray


# IO

def load_state():
    if not OUT_PATH.exists():
        return {}
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def norm_token(t: str) -> str:
    return t.strip().lower()


# FAISS / VECTOR ACCESS

def get_faiss_index(slice_range):
    if slice_range in _FAISS_CACHE:
        return _FAISS_CACHE[slice_range]

    index = FaissIndex.load(str(faiss_slice_path(slice_range)))
    _FAISS_CACHE[slice_range] = index
    return index


def get_vec_index(slice_id):
    if slice_id in _VEC_CACHE:
        return _VEC_CACHE[slice_id]

    vecs, id_to_pos, ids = load_id_vectors(slice_id)

    _VEC_CACHE[slice_id] = (vecs, id_to_pos, ids)
    return vecs, id_to_pos, ids


# DB

def get_token_occurrence_ids(conn, token, start, end):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT token_occurrence_id
            FROM pamphlet_tokens
            WHERE lower(token) = %s
              AND pub_year >= %s
              AND pub_year < %s
        """, (token, start, end))
        return [row[0] for row in cur.fetchall()]


def lookup_token_metadata(conn, ids):
    if not ids:
        return {}

    with conn.cursor() as cur:
        cur.execute("""
            SELECT token_occurrence_id, token, doc_id
            FROM pamphlet_tokens
            WHERE token_occurrence_id = ANY(%s)
        """, (ids,))
        rows = cur.fetchall()

    return {i: (t, d) for i, t, d in rows}


# OT

def kernel_weights(similarities, alpha):
    w = np.exp(alpha * (similarities - 1.0))
    s = np.sum(w)
    return w / s if s > 0 else np.ones_like(w) / len(w)


def wasserstein_drift(m1: OTMeasure, m2: OTMeasure) -> float:
    M = pairwise_distances(m1.X, m2.X, metric="cosine")
    return float(ot.sinkhorn2(m1.w, m2.w, M, reg=0.05))


# OT COMPRESSION (CRITICAL FOR D3)
def compress_measure(X, w, k=OT_CLUSTERS):
    if len(X) <= k:
        return X, w

    idx = np.random.choice(len(X), k, replace=False)
    centers = X[idx]

    for _ in range(5):
        dists = pairwise_distances(X, centers, metric="cosine")
        assign = dists.argmin(axis=1)

        new_centers = []
        new_weights = []

        for j in range(k):
            mask = assign == j
            if not np.any(mask):
                continue

            wj = w[mask]
            Xj = X[mask]

            wsum = wj.sum()
            center = (Xj * wj[:, None]).sum(axis=0) / wsum

            new_centers.append(center)
            new_weights.append(wsum)

        centers = np.array(new_centers)
        weights = np.array(new_weights)

    weights = weights / weights.sum()
    return centers, weights


# CORE
def compute_drift(token, conn, use_kernel, alpha):
    slices_data = []
    prev_measure = None

    for start, end in SLICES:
        sid = f"{start}-{end}"

        index = get_faiss_index((start, end))
        vecs, id_to_pos, _ = get_vec_index(sid)

        occ_ids = get_token_occurrence_ids(conn, token, start, end)

        positions = [id_to_pos[i] for i in occ_ids if i in id_to_pos]

        if not positions:
            slices_data.append(empty_slice(start, end))
            continue

        X_accum = []
        w_accum = []
        neighbor_mass = defaultdict(float)
        doc_ids = []

        for i, pos in enumerate(positions):
            vec = vecs[pos]

            D, I = index.search(vec.reshape(1, -1), K_NEIGHBORS * 5)
            neigh_ids = I[0]
            sims = D[0]

            meta = lookup_token_metadata(conn, neigh_ids.tolist())

            weights = kernel_weights(sims, alpha) if use_kernel else np.ones_like(sims) / len(sims)

            for nid, w in zip(neigh_ids, weights):
                if nid not in id_to_pos:
                    continue

                if nid not in meta:
                    continue

                tkn, doc = meta[nid]

                if tkn == token or tkn.lower() in STOPWORDS:
                    continue

                neigh_vec = vecs[id_to_pos[nid]]

                X_accum.append(neigh_vec)
                w_accum.append(float(w))

                neighbor_mass[tkn] += float(w)

                if i < 10:
                    doc_ids.append(doc)

        if not w_accum:
            curr_measure = None
            centroids, masses = [], []
        else:
            X = np.array(X_accum, dtype=np.float32)
            w = np.array(w_accum, dtype=np.float32)
            w /= w.sum()

            Xc, wc = compress_measure(X, w)

            curr_measure = OTMeasure(X=Xc, w=wc)

            centroids = Xc.tolist()
            masses = wc.tolist()

        drift = 0.0
        if prev_measure is not None and curr_measure is not None:
            drift = wasserstein_drift(prev_measure, curr_measure)

        prev_measure = curr_measure

        top_neighbors = sorted(
            [{"token": t, "mass": v} for t, v in neighbor_mass.items()],
            key=lambda x: -x["mass"]
        )[:TOP_K_NEIGHBORS]

        slices_data.append({
            "slice_start": start,
            "slice_end": end,
            "drift": drift,

            "corpus_count": len(positions),
            "support_count": len(X_accum),

            "top_neighbors": top_neighbors,
            "top_docs": Counter(doc_ids).most_common(5),

            # geometry for D3
            "centroids": centroids,
            "centroid_weights": masses,
        })

    return slices_data


def empty_slice(start, end):
    return {
        "slice_start": start,
        "slice_end": end,
        "drift": 0.0,
        "corpus_count": 0,
        "support_count": 0,
        "top_neighbors": [],
        "top_docs": [],
        "centroids": [],
        "centroid_weights": [],
    }


# PIPELINE
def update_token(token, conn, state, use_kernel, alpha):
    logger.info(f"token={token}")

    result = compute_drift(token, conn, use_kernel, alpha)

    state[token] = {
        f"{r['slice_start']}-{r['slice_end']}": r
        for r in result
    }

    return state


def run_batch(tokens, conn, state, use_kernel, alpha):
    for token in tokens:
        state = update_token(norm_token(token), conn, state, use_kernel, alpha)
    return state


# ENTRY
def main():
    conn = get_connection()

    parser = argparse.ArgumentParser()
    parser.add_argument("token", nargs="?", default=None)
    parser.add_argument("--no-kernel", action="store_true")
    parser.add_argument("--alpha", type=float, default=KERNEL_ALPHA_DEFAULT)

    args = parser.parse_args()
    use_kernel = USE_KERNEL_DEFAULT and not args.no_kernel

    state = load_state()

    if args.token:
        state = update_token(norm_token(args.token), conn, state, use_kernel, args.alpha)
    else:
        state = run_batch(list(CONCEPT_SETS.keys()), conn, state, use_kernel, args.alpha)

    save_state(state)

    print(json.dumps({
        "status": "ok",
        "tokens": len(state)
    }, indent=2))


if __name__ == "__main__":
    main()
