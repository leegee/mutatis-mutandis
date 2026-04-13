#!/usr/bin/env python

import json
import time
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


def select_neighbors(
    tokens,
    sims,
    *,
    top_k=15,
    min_sim=0.35,
    use_percentile=True,
    percentile=75.0,
):
    if len(tokens) != len(sims):
        raise ValueError("tokens and sims must be same length")

    if not tokens:
        return []

    sims_arr = np.array(sims, dtype=float)

    if use_percentile:
        cutoff = float(np.percentile(sims_arr, percentile))
        threshold = max(min_sim, cutoff)
    else:
        threshold = min_sim

    filtered = [
        (t, s)
        for t, s in zip(tokens, sims)
        if s >= threshold
    ]

    if not filtered:
        pairs = list(zip(tokens, sims))
        pairs.sort(key=lambda x: -x[1])
        return pairs[:top_k]

    filtered.sort(key=lambda x: -x[1])
    return filtered[:top_k]


def get_faiss_index(slice_range):
    if slice_range in _FAISS_CACHE:
        return _FAISS_CACHE[slice_range]

    path = str(faiss_slice_path(slice_range))
    index = FaissIndex.load(path)
    _FAISS_CACHE[slice_range] = index
    return index


def save_id_cache():
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _ID_CACHE.items()}, f)


def load_id_cache():
    if not _CACHE_FILE.exists():
        return
    with _CACHE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
        _ID_CACHE.update({int(k): tuple(v) for k, v in data.items()})


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


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def entropy_from_tokens(tokens):
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=float) / total
    return float(-np.sum(p * np.log(p)))


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


def match_clusters(prev_clusters, curr_clusters, threshold=0.25):
    if not prev_clusters:
        return [], list(range(len(curr_clusters))), []

    prev = np.array(prev_clusters)
    curr = np.array(curr_clusters)

    matches = []
    used_prev = set()

    for i, c in enumerate(curr):
        dists = [1 - cosine(c, p) for p in prev]
        j = int(np.argmin(dists))
        d = dists[j]

        if d <= threshold:
            matches.append((j, i, d))
            used_prev.add(j)
        else:
            matches.append((None, i, d))

    deaths = [j for j in range(len(prev)) if j not in used_prev]
    return matches, deaths


def compute_drift_and_neighbors_clustered(token, conn):
    slices_data = []
    prev_clusters = []
    prev_neighbor_counts = None

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
                "entropy": 0.0,
                "top_neighbors": [],
                "count": 0,
                "top_docs": [],
                "drift": 0.0,
                "births": 0,
                "deaths": 0,
                "js_divergence": 0.0
            })
            continue

        neighbor_tokens = []
        neighbor_sims = []
        doc_ids = []

        for i, vec in enumerate(vecs):
            distances, neighbor_ids = index.search(vec.reshape(1, -1), K_NEIGHBORS * 5)
            rows = lookup_token_occurrences(conn, neighbor_ids[0].tolist())

            if i < 10:
                doc_ids.extend([doc for (_, _, doc, _) in rows])

            for (tkn, _can, _doc, _year), sim in zip(rows, distances[0], strict=True):
                if tkn == token:
                    continue
                neighbor_tokens.append(tkn)
                neighbor_sims.append(sim)

        pairs = [
            (t, s)
            for t, s in zip(neighbor_tokens, neighbor_sims)
            if t.lower() not in STOPWORDS
        ]

        tokens_, sims_ = zip(*pairs) if pairs else ([], [])

        selected = select_neighbors(
            list(tokens_),
            list(sims_),
            top_k=TOP_K_NEIGHBORS,
            min_sim=0.35,
            use_percentile=True,
            percentile=75.0
        )

        neighbor_map = defaultdict(lambda: {"count": 0, "sim_sum": 0.0})

        for t, s in selected:
            neighbor_map[t]["count"] += 1
            neighbor_map[t]["sim_sum"] += s

        curr_counts = Counter({t: v["count"] for t, v in neighbor_map.items()})

        top_neighbors = [
            {
                "token": t,
                "count": v["count"],
                "similarity": float(v["sim_sum"] / v["count"])
            }
            for t, v in neighbor_map.items()
        ]

        top_neighbors.sort(key=lambda x: -x["similarity"])
        top_neighbors = top_neighbors[:TOP_K_NEIGHBORS]

        ent = entropy_from_tokens([t for t, _ in selected])

        slice_entry = {
            "slice_start": start,
            "slice_end": end,
            "n_clusters": len(cluster_centroids),
            "cluster_sizes": cluster_sizes,
            "entropy": ent,
            "top_neighbors": top_neighbors,
            "count": int(sum(cluster_sizes)) if cluster_sizes else 0,
            "top_docs": Counter(doc_ids).most_common(5) if doc_ids else [],
            "drift": 0.0,
            "births": 0,
            "deaths": 0,
            "js_divergence": 0.0
        }

        if prev_clusters:
            matches, deaths = match_clusters(prev_clusters, cluster_centroids)

            movements = [d for p, c, d in matches if p is not None]
            drift_val = float(np.mean(movements)) if movements else 1.0
            births = sum(1 for p, c, d in matches if p is None)

            j = js_divergence(prev_neighbor_counts, curr_counts)

            slice_entry.update({
                "drift": drift_val,
                "births": births,
                "deaths": len(deaths),
                "js_divergence": j
            })

        prev_neighbor_counts = curr_counts
        prev_clusters = cluster_centroids

        slices_data.append(slice_entry)

    slices_data.sort(key=lambda s: s["slice_start"])
    return slices_data
