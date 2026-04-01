#!/usr/bin/env python

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import json
from pathlib import Path

from mb_embedding_pipeline import load_vectors
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR
import psycopg

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
    sid = f"{start}-{end}"
    path = str(faiss_slice_path(slice_range))

    logger.info(f"[{sid}] loading FAISS index from disk: {path}")
    t0 = time.time()

    index = FaissIndex.load(path)

    logger.info(f"[{sid}] FAISS loaded in {time.time() - t0:.2f}s (cached)")
    _FAISS_CACHE[slice_range] = index
    return index


# ID lookup cache - might save time when running many queries
_ID_CACHE: dict[int, tuple[str, str | None, str, int]] = {}

def save_id_cache():
    with _CACHE_FILE.open("w", encoding="utf-8") as f:
        # Convert keys to str for JSON
        json.dump({str(k): v for k, v in _ID_CACHE.items()}, f)
    logger.info(f"Saved ID cache to {_CACHE_FILE} ({len(_ID_CACHE)} entries)")

def load_id_cache():
    if not _CACHE_FILE.exists():
        logger.info(f"No cache file found at {_CACHE_FILE}, starting empty")
        return
    with _CACHE_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
        # Convert keys back to int
        _ID_CACHE.update({int(k): tuple(v) for k, v in data.items()})
    logger.info(f"Loaded ID cache from {_CACHE_FILE} ({len(_ID_CACHE)} entries)")

def lookup_token_occurrences(conn: psycopg.Connection, ids: list[int]):
    if not ids:
        return []

    t0 = time.time()
    missing = [i for i in ids if i not in _ID_CACHE]

    if missing:
        logger.info(f"[lookup] cache miss: {len(missing)}/{len(ids)} ids")

        t_db = time.time()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT token_occurrence_id, token, canonical, doc_id, pub_year
                FROM pamphlet_tokens
                WHERE token_occurrence_id = ANY(%s)
            """, (missing,))
            rows = cur.fetchall()

        for occ_id, token, canonical, doc_id, year in rows:
            _ID_CACHE[occ_id] = (token, canonical, doc_id, year)

        logger.info(
            f"[lookup] fetched {len(rows)} rows in {time.time() - t_db:.2f}s "
            f"(cache size={len(_ID_CACHE)})"
        )

        if len(rows) != len(missing):
            logger.warning(
                f"[lookup] DB returned {len(rows)} rows for {len(missing)} ids"
            )
    else:
        logger.info(f"[lookup] cache hit: {len(ids)} ids")

    result = [_ID_CACHE[i] for i in ids if i in _ID_CACHE]

    logger.info(f"[lookup] total {time.time() - t0:.2f}s")
    return result


# Metrics
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

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def compute_centroid(slice_id: str, token: str):
    data = load_vectors(slice_id)
    vecs = data.get(token, [])

    if not vecs:
        logger.info(f"[{slice_id}] token='{token}' - no occurrences")
        return None, []

    centroid = np.mean(np.stack(vecs), axis=0)
    logger.info(f"[{slice_id}] token='{token}' n={len(vecs)}")
    return centroid, vecs


def slice_neighbors(conn, index, centroid_vec, slice_id):
    distances, neighbor_ids = index.search(
        centroid_vec.reshape(1, -1),
        k=K_NEIGHBORS + 1
    )

    neighbor_ids = neighbor_ids[0][1:]
    distances = distances[0][1:]

    rows = lookup_token_occurrences(conn, neighbor_ids.tolist())

    tokens = []
    agg = defaultdict(lambda: {"freq": 0, "sim_sum": 0.0})

    for (tkn, can, doc, year), dist in zip(rows, distances):
        tokens.append(tkn)
        key = (tkn, can)
        agg[key]["freq"] += 1
        agg[key]["sim_sum"] += float(dist)

    top_neighbors = []
    for (tkn, can), info in agg.items():
        top_neighbors.append({
            "token": tkn,
            "canonical": can,
            "freq": info["freq"],
            "mean_sim": info["sim_sum"] / info["freq"],
        })

    top_neighbors.sort(key=lambda x: x["mean_sim"], reverse=True)

    return top_neighbors[:K_NEIGHBORS], tokens


def compute_drift_and_neighbors(token: str, conn):
    centroids = []
    slice_years = []
    neighbors_per_slice = []
    entropy_per_slice = []
    token_dists = []

    for (start, end) in SLICES:
        sid = f"{start}-{end}"

        centroid, _ = compute_centroid(sid, token)
        if centroid is None:
            continue

        centroids.append(centroid)
        slice_years.append(start)

        index = get_faiss_index((start, end))
        top_neighbors, tokens = slice_neighbors(conn, index, centroid, sid)

        neighbors_per_slice.append(top_neighbors)

        # spread
        ent = entropy_from_tokens(tokens)
        entropy_per_slice.append(ent)

        token_dists.append(Counter(tokens))

        logger.info(f"[{sid}] entropy={ent:.4f}")

    if len(centroids) < 2:
        return [], [], [], [], []

    drifts = []
    drift_x = []
    jsd = []

    for i in range(1, len(centroids)):
        d = 1 - cosine(centroids[i], centroids[i - 1])
        drifts.append(float(d))
        drift_x.append(slice_years[i])

        j = js_divergence(token_dists[i - 1], token_dists[i])
        jsd.append(j)

        logger.info(
            f"drift {slice_years[i-1]} to {slice_years[i]}={d:.4f} "
            f"jsd={j:.4f}"
        )

    return drift_x, drifts, neighbors_per_slice, entropy_per_slice, jsd


def main():
    logger.info("Starting drift + neighborhood analysis")

    load_id_cache()

    conn = get_connection()
    plt.figure(figsize=(12, 6))

    terms = ['king']
    results = {}

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing '{token}'")

        x, drift, neighbors, entropy_vals, jsd_vals = \
            compute_drift_and_neighbors(token, conn)

        if not drift:
            continue

        save_id_cache()

        results[token] = {
            "years": x,
            "drift": drift,
            "entropy": entropy_vals,
            "js_divergence": jsd_vals,
            "neighbors": neighbors,
        }

        plt.plot(x, drift, marker='o', label=f"{token} drift")

    out_path = OUT_DIR / "drift_neighbors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved dataset to {out_path}")

    plt.xlabel("Year")
    plt.ylabel("Drift")
    plt.title("Semantic Drift + Distributional Change")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
