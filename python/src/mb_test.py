#!/usr/bin/env python

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.cluster import KMeans

from mb_embedding_pipeline import load_vectors
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR
import psycopg
import hdbscan

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

# FAISS cache
_FAISS_CACHE: dict[tuple[int, int], FaissIndex] = {}

_CACHE_FILE = OUT_DIR / "token_occurrence_cache.json"

def get_faiss_index(slice_range: tuple[int, int]) -> FaissIndex:
    if slice_range in _FAISS_CACHE:
        logger.info(f"[{slice_range}] FAISS cache hit")
        return _FAISS_CACHE[slice_range]

    start, end = slice_range
    path = str(faiss_slice_path(slice_range))
    logger.info(f"[{start}-{end}] loading FAISS index from disk: {path}")
    t0 = time.time()

    index = FaissIndex.load(path)
    logger.info(f"[{start}-{end}] FAISS loaded in {time.time() - t0:.2f}s (cached)")
    _FAISS_CACHE[slice_range] = index
    return index

# ID lookup cache
_ID_CACHE: dict[int, tuple[str, str | None, str, int]] = {}

def save_id_cache():
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _ID_CACHE.items()}, f)
    logger.info(f"Saved ID cache to {_CACHE_FILE} ({len(_ID_CACHE)} entries)")

def load_id_cache():
    if not _CACHE_FILE.exists():
        logger.info(f"No cache file found at {_CACHE_FILE}, starting empty")
        return
    with _CACHE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
        _ID_CACHE.update({int(k): tuple(v) for k, v in data.items()})
    logger.info(f"Loaded ID cache from {_CACHE_FILE} ({len(_ID_CACHE)} entries)")

def lookup_token_occurrences(conn: psycopg.Connection, ids: list[int]):
    if not ids:
        return []

    t0 = time.time()
    missing = [i for i in ids if i not in _ID_CACHE]

    if missing:
        logger.info(f"[lookup] cache miss: {len(missing)}/{len(ids)} ids")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token_occurrence_id, token, canonical, doc_id, pub_year
                FROM pamphlet_tokens
                WHERE token_occurrence_id = ANY(%s)
            """, (missing,))
            rows = cur.fetchall()
        for occ_id, token, canonical, doc_id, year in rows:
            _ID_CACHE[occ_id] = (token, canonical, doc_id, year)
        logger.info(f"[lookup] fetched {len(rows)} rows in {time.time() - t0:.2f}s (cache size={len(_ID_CACHE)})")

    return [_ID_CACHE[i] for i in ids if i in _ID_CACHE]

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def entropy_from_tokens(tokens: list[str]) -> float:
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    p = np.array(list(counts.values()), dtype=float) / total
    return float(-np.sum(p * np.log(p)))

def js_divergence(p_counts: Counter, q_counts: Counter) -> float:
    vocab = set(p_counts) | set(q_counts)
    p = np.array([p_counts.get(t, 0) for t in vocab], dtype=float)
    q = np.array([q_counts.get(t, 0) for t in vocab], dtype=float)

    if p.sum() == 0 or q.sum() == 0:
        return 0.0

    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    mask_p = p > 0
    mask_q = q > 0
    kl = lambda a, b, mask: np.sum(a[mask] * np.log(a[mask] / b[mask]))
    return float(0.5 * kl(p, m, mask_p) + 0.5 * kl(q, m, mask_q))

def compute_clusters(slice_id: str, token: str):
    data = load_vectors(slice_id)
    vecs = data.get(token, [])
    if not vecs:
        return [], []

    vecs = np.array(vecs, dtype=np.float32)

    # already normalized upstream — do NOT renormalize

    n = len(vecs)

    # small-n fallback
    if n < 8:
        return [np.mean(vecs, axis=0)], vecs

    # heuristic: 2–3 senses max
    k = min(3, max(2, n // 100))  # grows slowly with data

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(vecs)

    clusters = []
    for i in range(k):
        members = vecs[labels == i]
        if len(members) == 0:
            continue
        clusters.append(np.mean(members, axis=0))

    return clusters, vecs


def compute_drift_and_neighbors_clustered(token: str, conn):
    slice_years = []
    clusters_per_slice = []
    entropy_per_slice = []
    token_dists = []

    for start, end in SLICES:
        sid = f"{start}-{end}"
        index = get_faiss_index((start, end))
        cluster_centroids, vecs = compute_clusters(sid, token)

        if not cluster_centroids:
            logger.info(f"[{sid}] token='{token}' clusters=0 entropy=0.0")
            continue

        slice_years.append(start)
        clusters_per_slice.append(cluster_centroids)

        # local neighborhoods
        neighbor_tokens = []
        for vec in vecs:
            distances, neighbor_ids = index.search(vec.reshape(1, -1), K_NEIGHBORS * 5)
            rows = lookup_token_occurrences(conn, neighbor_ids[0].tolist())
            for (tkn, can, doc, year), sim in zip(rows, distances[0]):
                if sim < 0.6 or tkn == token:
                    continue
                neighbor_tokens.append(tkn)

        ent = entropy_from_tokens(neighbor_tokens)
        entropy_per_slice.append(ent)
        token_dists.append(Counter(neighbor_tokens))

        logger.info(f"[{sid}] clusters={len(cluster_centroids)} entropy={ent:.4f}")

    if len(clusters_per_slice) < 2:
        return [], [], [], [], []

    drift_x = []
    drifts = []
    jsd = []

    for i in range(1, len(clusters_per_slice)):
        # match clusters across slices by nearest neighbors
        prev = clusters_per_slice[i - 1]
        curr = clusters_per_slice[i]

        # compute drift as average nearest-cluster distance
        distances = [min([1 - cosine(c, p) for p in prev]) for c in curr]
        drift_val = float(np.mean(distances))
        drifts.append(drift_val)
        drift_x.append(slice_years[i])

        j = js_divergence(token_dists[i - 1], token_dists[i])
        jsd.append(j)
        logger.info(f"drift {slice_years[i-1]}->{slice_years[i]}={drift_val:.4f} jsd={j:.4f}")

    return drift_x, drifts, clusters_per_slice, entropy_per_slice, jsd

def main():
    logger.info("Starting cluster-aware drift + neighborhood analysis")
    load_id_cache()
    conn = get_connection()
    plt.figure(figsize=(12, 6))

    terms = ['liberty']
    results = {}

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing '{token}'")
        x, drift, clusters, entropy_vals, jsd_vals = compute_drift_and_neighbors_clustered(token, conn)
        if not drift:
            continue

        save_id_cache()
        results[token] = {
            "years": x,
            "drift": drift,
            "entropy": entropy_vals,
            "js_divergence": jsd_vals,
            "clusters": [[c.tolist() for c in slice_clusters] for slice_clusters in clusters]
        }

        plt.plot(x, drift, marker='o', label=f"{token} drift")

    out_path = OUT_DIR / "drift_neighbors_clustered.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved dataset to {out_path}")

    plt.xlabel("Year")
    plt.ylabel("Drift")
    plt.title("Cluster-aware Semantic Drift + Distributional Change")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
