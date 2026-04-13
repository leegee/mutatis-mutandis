#!/usr/bin/env python

import json
import numpy as np
from collections import Counter, defaultdict
import hdbscan

from mb_embedding_pipeline import load_vectors
from lib.js_divergence import js_divergence
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR, CONCEPT_SETS, SLICES
from lib.wordlist import STOPWORDS

MIN_TOKENS = 30

OUT_PATH = OUT_DIR / "drift_neighbors_micro_senses_slices.json"

K_NEIGHBORS = 5
TOP_K_NEIGHBORS = 15

_FAISS_CACHE = {}
_CACHE_FILE = OUT_DIR / "token_occurrence_cache.json"
_ID_CACHE = {}


# ----------------------------
# FAISS cache
# ----------------------------
def get_faiss_index(slice_range):
    if slice_range in _FAISS_CACHE:
        return _FAISS_CACHE[slice_range]

    path = str(faiss_slice_path(slice_range))
    index = FaissIndex.load(path)
    _FAISS_CACHE[slice_range] = index
    return index


# ----------------------------
# token lookup cache
# ----------------------------
def lookup_token_occurrences(conn, ids):
    if not ids:
        return []

    missing = [i for i in ids if i not in _ID_CACHE]

    if missing:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token_occurrence_id, token, canonical, doc_id, pub_year
                FROM pamphlet_tokens
                WHERE token_occurrence_id = ANY(%s)
            """, (missing,))
            rows = cur.fetchall()

        for occ_id, token, canonical, doc_id, year in rows:
            _ID_CACHE[occ_id] = (token, canonical, doc_id, year)

    return [_ID_CACHE[i] for i in ids if i in _ID_CACHE]


# ----------------------------
# entropy
# ----------------------------
def entropy_from_tokens(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0

    p = np.array(list(counts.values()), dtype=float) / total
    return float(-np.sum(p * np.log(p)))


# ----------------------------
# normalize distribution
# ----------------------------
def normalize_counts(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


# ----------------------------
# cluster inference (unchanged)
# ----------------------------
def compute_clusters(slice_id, token):
    data = load_vectors(slice_id)
    vecs = data.get(token, [])

    if not vecs:
        return [], [], []

    vecs = np.array(vecs, dtype=np.float32)
    n = len(vecs)

    if n < 15:
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm != 0:
            centroid = centroid / norm
        return [centroid], vecs, [n]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(5, n // 200),
        min_samples=None
    )

    labels = clusterer.fit_predict(vecs)

    clusters = []
    sizes = []

    for label in set(labels):
        if label == -1:
            continue

        members = vecs[labels == label]
        if len(members) > 0:
            centroid = np.mean(members, axis=0)
            norm = np.linalg.norm(centroid)
            if norm == 0:
                continue
            centroid = centroid / norm

            clusters.append(centroid)
            sizes.append(len(members))

    if not clusters:
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm != 0:
            centroid = centroid / norm
        return [centroid], vecs, [n]

    return clusters, vecs, sizes


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def compute_drift_and_neighbors_clustered(token, conn):
    slices_data = []

    prev_neighbor_dist = None

    for start, end in SLICES:
        sid = f"{start}-{end}"
        index = get_faiss_index((start, end))

        cluster_centroids, vecs, cluster_sizes = compute_clusters(sid, token)

        if not cluster_centroids:
            slices_data.append({
                "slice_start": start,
                "slice_end": end,
                "n_clusters": 0,
                "cluster_sizes": [],

                "corpus_count": 0,
                "support_count": 0,

                "entropy": 0.0,
                "top_neighbors": [],
                "top_docs": [],

                "drift": 0.0,
                "js_divergence": 0.0
            })
            continue

        # ----------------------------
        # retrieval-space accumulation
        # ----------------------------
        retrieval_map = defaultdict(lambda: {"support_count": 0, "sim_sum": 0.0})

        # corpus-space approximation within slice context
        corpus_counts = Counter()
        doc_ids = []

        for i, vec in enumerate(vecs):
            distances, neighbor_ids = index.search(
                vec.reshape(1, -1),
                K_NEIGHBORS * 5
            )

            rows = lookup_token_occurrences(
                conn,
                neighbor_ids[0].tolist()
            )

            if i < 10:
                doc_ids.extend([doc for (_, _, doc, _) in rows])

            for (tkn, _can, doc, _year), sim in zip(rows, distances[0]):
                if tkn == token:
                    continue
                if tkn.lower() in STOPWORDS:
                    continue

                corpus_counts[tkn] += 1

                m = retrieval_map[tkn]
                m["support_count"] += 1
                m["sim_sum"] += float(sim)

        # ----------------------------
        # distributions
        # ----------------------------
        retrieval_counts = Counter({
            t: v["support_count"]
            for t, v in retrieval_map.items()
        })

        curr_dist = normalize_counts(retrieval_counts)

        # ----------------------------
        # entropy (retrieval space)
        # ----------------------------
        ent = entropy_from_tokens(list(retrieval_counts.elements()))

        # ----------------------------
        # scored neighbors
        # ----------------------------
        scored_neighbors = []

        for t, v in retrieval_map.items():
            support_count = v["support_count"]
            sim_sum = v["sim_sum"]

            score = sim_sum / np.sqrt(support_count) if support_count > 0 else 0.0

            scored_neighbors.append({
                "token": t,
                "support_count": support_count,
                "similarity": float(score)
            })

        scored_neighbors.sort(key=lambda x: -x["similarity"])
        top_neighbors = scored_neighbors[:TOP_K_NEIGHBORS]

        # ----------------------------
        # JS divergence drift
        # ----------------------------
        drift_val = 0.0
        if prev_neighbor_dist is not None:
            drift_val = float(js_divergence(prev_neighbor_dist, curr_dist))

        # ----------------------------
        # slice output
        # ----------------------------
        slice_entry = {
            "slice_start": start,
            "slice_end": end,
            "n_clusters": len(cluster_centroids),
            "cluster_sizes": cluster_sizes,

            # corpus space (approx within FAISS retrieval context)
            "corpus_count": int(sum(corpus_counts.values())),

            # retrieval space
            "support_count": int(sum(retrieval_counts.values())),

            "entropy": ent,
            "top_neighbors": top_neighbors,
            "top_docs": Counter(doc_ids).most_common(5) if doc_ids else [],

            # drift in unified retrieval probability space
            "drift": drift_val,
            "js_divergence": drift_val
        }

        prev_neighbor_dist = curr_dist
        slices_data.append(slice_entry)

    slices_data.sort(key=lambda s: s["slice_start"])
    return slices_data
