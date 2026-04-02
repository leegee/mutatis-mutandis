#!/usr/bin/env python

import json
import time
import numpy as np
from collections import Counter
from pathlib import Path

from mb_embedding_pipeline import load_vectors
from lib.js_divergence import js_divergence
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR
import psycopg
import hdbscan

MIN_TOKENS = 30

SLICES = [
    (1625, 1629),
    (1630, 1634),
    (1635, 1639),
    (1640, 1640),
    (1641, 1641),
    (1642, 1642),
    (1643, 1643),
    (1644, 1644),
    (1645, 1645),
    (1646, 1646),
    (1647, 1647),
    (1648, 1648),
]

K_NEIGHBORS = 5
TOP_K_NEIGHBORS = 15

_FAISS_CACHE = {}
_CACHE_FILE = OUT_DIR / "token_occurrence_cache.json"
_ID_CACHE = {}


def get_faiss_index(slice_range):
    if slice_range in _FAISS_CACHE:
        logger.info(f"[{slice_range}] FAISS cache hit")
        return _FAISS_CACHE[slice_range]

    path = str(faiss_slice_path(slice_range))
    logger.debug(f"[{slice_range}] loading FAISS index: {path}")
    t0 = time.time()

    index = FaissIndex.load(path)

    logger.debug(f"[{slice_range}] FAISS loaded in {time.time() - t0:.2f}s")
    _FAISS_CACHE[slice_range] = index
    return index


def save_id_cache():
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _ID_CACHE.items()}, f)
    logger.info(f"Saved ID cache ({len(_ID_CACHE)} entries)")


def load_id_cache():
    if not _CACHE_FILE.exists():
        logger.info("No ID cache found, starting fresh")
        return
    with _CACHE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
        _ID_CACHE.update({int(k): tuple(v) for k, v in data.items()})
    logger.info(f"Loaded ID cache ({len(_ID_CACHE)} entries)")


def lookup_token_occurrences(conn, ids):
    if not ids:
        return []

    missing = [i for i in ids if i not in _ID_CACHE]

    if missing:
        logger.debug(f"[lookup] cache miss: {len(missing)}/{len(ids)}")
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

    # clustering unstable for very small n
    if n < 15:
        return [np.mean(vecs, axis=0)], vecs, [n]

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
            clusters.append(np.mean(members, axis=0))
            sizes.append(len(members))

    # failure: all noise → collapse
    if not clusters:
        return [np.mean(vecs, axis=0)], vecs, [n]

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


def detect_phase_transitions(slices_data):
    if len(slices_data) < 3:
        return []

    drifts = np.array([s["drift"] for s in slices_data[1:]])
    jsd = np.array([s["js_divergence"] for s in slices_data[1:]])
    births = np.array([s["births"] for s in slices_data[1:]])

    def zscore(x):
        if np.std(x) == 0:
            return np.zeros_like(x)
        return (x - np.mean(x)) / np.std(x)

    drift_z = zscore(drifts)
    jsd_z = zscore(jsd)
    births_z = zscore(births)

    score = drift_z + jsd_z + 0.5 * births_z

    transitions = []

    for i, s in enumerate(score):
        if s > 1.5 and drift_z[i] > 0.5 and jsd_z[i] > 0.5:
            transitions.append({
                "year": slices_data[i + 1]["year"],
                "score": float(s),
                "drift": slices_data[i + 1]["drift"],
                "js_divergence": slices_data[i + 1]["js_divergence"],
                "births": slices_data[i + 1]["births"],
                "deaths": slices_data[i + 1]["deaths"]
            })

    return transitions


def compute_drift_and_neighbors_clustered(token, conn):
    slices_data = []
    prev_clusters = []
    prev_neighbor_counts = None
    prev_year = None

    for start, end in SLICES:
        sid = f"{start}-{end}"
        index = get_faiss_index((start, end))

        cluster_centroids, vecs, cluster_sizes = compute_clusters(sid, token)

        if not cluster_centroids:
            continue

        neighbor_tokens = []

        for vec in vecs:
            distances, neighbor_ids = index.search(vec.reshape(1, -1), K_NEIGHBORS * 5)
            rows = lookup_token_occurrences(conn, neighbor_ids[0].tolist())

            for (tkn, can, doc, year), sim in zip(rows, distances[0]):
                if sim < 0.6 or tkn == token:
                    continue
                neighbor_tokens.append(tkn)

        counts = Counter(neighbor_tokens)
        top_neighbors = counts.most_common(TOP_K_NEIGHBORS)

        logger.info(
            f"[{sid}] token='{token}' "
            f"clusters={len(cluster_centroids)} sizes={cluster_sizes} "
            f"entropy={entropy_from_tokens(neighbor_tokens):.4f} "
            f"neighbors={len(neighbor_tokens)}"
        )

        logger.info(
            f"[{sid}] top_neighbors="
            + ", ".join(f"{w}:{c}" for w, c in top_neighbors[:8])
        )

        ent = entropy_from_tokens(neighbor_tokens)

        slice_entry = {
            "year": start,
            "n_clusters": len(cluster_centroids),
            "cluster_sizes": cluster_sizes,
            "entropy": ent,
            "top_neighbors": top_neighbors,
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

            curr_counts = counts
            j = js_divergence(prev_neighbor_counts, curr_counts)

            logger.info(
                f"{prev_year}->{start} "
                f"drift={drift_val:.4f} births={births} deaths={len(deaths)} "
                f"jsd={j:.4f}"
            )

            slice_entry.update({
                "drift": drift_val,
                "births": births,
                "deaths": len(deaths),
                "js_divergence": j
            })

            prev_neighbor_counts = curr_counts
        else:
            prev_neighbor_counts = counts

        prev_clusters = cluster_centroids
        prev_year = start

        slices_data.append(slice_entry)

    return slices_data


def main():
    logger.info("Starting cluster-aware drift + phase transition detection")

    load_id_cache()
    conn = get_connection()

    terms = ['liberty', 'freedom']
    results = {}

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing '{token}'")

        slices_data = compute_drift_and_neighbors_clustered(token, conn)

        if not slices_data:
            continue

        transitions = detect_phase_transitions(slices_data)

        logger.info(f"[{token}] detected {len(transitions)} phase transitions")

        for t in transitions:
            logger.info(
                f"[{token}] PHASE SHIFT @ {t['year']} "
                f"(score={t['score']:.2f}, drift={t['drift']:.4f}, "
                f"jsd={t['js_divergence']:.4f}, births={t['births']})"
            )

        results[token] = {
            "slices": slices_data,
            "phase_transitions": transitions
        }

    save_id_cache()

    out_path = OUT_DIR / "drift_neighbors_micro_senses_slices.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
