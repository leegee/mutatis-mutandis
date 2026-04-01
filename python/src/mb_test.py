#!/usr/bin/env python

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from mb_embedding_pipeline import load_vectors
from lib.eebo_config import CONCEPT_SETS, OUT_DIR
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
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


# invariant: one FAISS index per slice; safe to reuse across tokens
_FAISS_CACHE: dict[tuple[int, int], FaissIndex] = {}


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

    logger.info(
        f"[{sid}] FAISS index loaded in {time.time() - t0:.2f}s (cached)"
    )

    _FAISS_CACHE[slice_range] = index
    return index


def warm_faiss_cache():
    logger.info("Warming FAISS cache for all slices")
    for slice_range in SLICES:
        get_faiss_index(slice_range)
    logger.info(f"FAISS cache warm: {len(_FAISS_CACHE)} indexes loaded")


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_centroid(slice_id: str, token: str):
    t0 = time.time()

    data = load_vectors(slice_id)
    vecs = data.get(token, [])

    if not vecs:
        logger.info(f"[{slice_id}] token='{token}' → no occurrences")
        return None, []

    centroid = np.mean(np.stack(vecs), axis=0)

    logger.info(
        f"[{slice_id}] token='{token}' → n={len(vecs)} "
        f"(centroid computed in {time.time() - t0:.2f}s)"
    )
    return centroid, vecs


def slice_neighbors(conn: psycopg.Connection, index: FaissIndex, centroid_vec: np.ndarray, slice_id: str):
    t0 = time.time()

    distances, neighbor_ids = index.search(
        centroid_vec.reshape(1, -1),
        k=K_NEIGHBORS + 1
    )

    neighbor_ids = neighbor_ids[0][1:]
    distances = distances[0][1:]

    if len(neighbor_ids) == 0:
        logger.warning(f"[{slice_id}] no neighbors returned from FAISS")
        return []

    logger.info(
        f"[{slice_id}] FAISS returned {len(neighbor_ids)} neighbors "
        f"in {time.time() - t0:.2f}s"
    )

    t1 = time.time()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT token, canonical, doc_id, pub_year
            FROM pamphlet_tokens
            WHERE token_occurrence_id = ANY(%s)
        """, (neighbor_ids.tolist(),))
        rows = cur.fetchall()

    logger.info(
        f"[{slice_id}] DB lookup returned {len(rows)} rows "
        f"in {time.time() - t1:.2f}s"
    )

    # invariant: rows and distances must align positionally
    if len(rows) != len(distances):
        logger.error(
            f"[{slice_id}] mismatch rows={len(rows)} distances={len(distances)}"
        )

    agg = defaultdict(lambda: {"freq": 0, "sim_sum": 0.0})

    for row, dist in zip(rows, distances):
        tkn, can, doc, year = row
        key = (tkn, can)
        agg[key]["freq"] += 1
        agg[key]["sim_sum"] += float(dist)

    top_neighbors = []
    for (tkn, can), info in agg.items():
        mean_sim = info["sim_sum"] / info["freq"]
        top_neighbors.append({
            "token": tkn,
            "canonical": can,
            "freq": info["freq"],
            "mean_sim": mean_sim,
        })

    top_neighbors.sort(key=lambda x: x["mean_sim"], reverse=True)

    logger.info(
        f"[{slice_id}] aggregated to {len(top_neighbors)} unique neighbors "
        f"(total time {time.time() - t0:.2f}s)"
    )

    return top_neighbors[:K_NEIGHBORS]


def compute_drift_and_neighbors(token: str, conn: psycopg.Connection):
    centroids = []
    slice_years = []
    neighbors_per_slice = []

    for i, (start, end) in enumerate(SLICES):
        sid = f"{start}-{end}"
        logger.info(f"[{sid}] ({i+1}/{len(SLICES)}) starting slice")

        centroid, vecs = compute_centroid(sid, token)

        if centroid is None:
            continue

        centroids.append(centroid)
        slice_years.append(start)

        index = get_faiss_index((start, end))

        top_neighbors = slice_neighbors(conn, index, centroid, sid)
        neighbors_per_slice.append(top_neighbors)

    if len(centroids) < 2:
        logger.warning(
            f"token='{token}' insufficient data for drift (n={len(centroids)})"
        )
        return [], [], []

    drifts = []
    drift_x = []

    for i in range(1, len(centroids)):
        d = 1 - cosine(centroids[i], centroids[i - 1])
        drifts.append(float(d))
        drift_x.append(slice_years[i])

        logger.info(
            f"token='{token}' drift {slice_years[i-1]}→{slice_years[i]} = {d:.4f}"
        )

    return drift_x, drifts, neighbors_per_slice


def main():
    logger.info("Starting Heuser-style drift + neighbors computation")

    t_global = time.time()

    conn = get_connection()
    plt.figure(figsize=(12, 6))

    # optional: front-load I/O cost and eliminate runtime stalls
    warm_faiss_cache()

    terms = ['king']

    results = {}

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing concept='{concept}' token='{token}'")

        t0 = time.time()

        x, y, neighbors_per_slice = compute_drift_and_neighbors(token, conn)

        if not y:
            logger.warning(f"Skipping concept='{concept}' (no drift data)")
            continue

        results[token] = {
            "years": x,
            "drift": y,
            "neighbors": neighbors_per_slice,
        }

        logger.info(
            f"concept='{concept}' completed in {time.time() - t0:.2f}s"
        )

        for year, top_neighbors in zip(x, neighbors_per_slice[1:]):
            logger.info(f"[{token}] slice {year} top neighbors:")
            for n in top_neighbors:
                logger.info(
                    f"  {n['token']} (canonical={n['canonical']}) "
                    f"freq={n['freq']} mean_sim={n['mean_sim']:.4f}"
                )

        plt.plot(x, y, marker='o', label=concept)

    out_path = OUT_DIR / "drift_neighbors.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved dataset → {out_path}")

    plt.xlabel("Year (start of slice)")
    plt.ylabel("Drift (1 - cosine)")
    plt.title("Heuser-style Token Drift + Top Semantic Neighbors")
    plt.legend()
    plt.tight_layout()

    logger.info("Rendering plot")
    plt.show()

    logger.info(
        f"Done (total runtime {time.time() - t_global:.2f}s)"
    )


if __name__ == "__main__":
    main()
