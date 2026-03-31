#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from mb_embedding_pipeline import load_vectors
from lib.eebo_config import CONCEPT_SETS
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
    (1649, 1649),
    (1650, 1650),
    (1651, 1651),
    (1652, 1654),
    (1655, 1657),
    (1658, 1660),
    # (1661, 1665),
]


K_NEIGHBORS = 5  # Number of semantic neighbors to retrieve

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_centroid(slice_id, token):
    data = load_vectors(slice_id)
    vecs = data.get(token, [])

    if not vecs:
        logger.debug(f"[{slice_id}] token='{token}' → no occurrences")
        return None, []

    logger.debug(f"[{slice_id}] token='{token}' → n={len(vecs)}")
    return np.mean(np.stack(vecs), axis=0), vecs

def get_neighbors(conn: psycopg.Connection, index: FaissIndex, vecs: list[np.ndarray]):
    if not vecs:
        return []

    vecs_arr = np.stack(vecs).astype(np.float32)
    distances, neighbor_ids = index.search(vecs_arr, k=K_NEIGHBORS + 1)  # +1 for self

    all_neighbors = []

    # Flatten and deduplicate neighbor IDs for one DB query
    neighbor_ids_flat = np.unique(neighbor_ids.flatten())
    neighbor_ids_flat = neighbor_ids_flat[neighbor_ids_flat != -1]  # FAISS uses -1 for missing

    if len(neighbor_ids_flat) == 0:
        return [[] for _ in vecs]

    with conn.cursor() as cur:
        cur.execute("""
            SELECT token_occurrence_id, token, canonical, doc_id, pub_year
            FROM pamphlet_tokens
            WHERE token_occurrence_id = ANY(%s)
        """, (neighbor_ids_flat.tolist(),))
        rows = cur.fetchall()

    id_to_info = {row[0]: row[1:] for row in rows}

    # Build neighbor lists per occurrence, skipping self
    for i in range(len(vecs)):
        occ_neighbors = []
        for dist, nid in zip(distances[i], neighbor_ids[i]):
            if nid == -1 or nid == -1:  # skip missing
                continue
            if nid in id_to_info:
                tkn, can, doc, year = id_to_info[nid]
                occ_neighbors.append(((tkn, can, doc, year), dist))
        # remove self-match (assumes first entry is self)
        occ_neighbors = [n for n in occ_neighbors if n[0][2] != -1]  # crude filter; refine if needed
        all_neighbors.append(occ_neighbors[:K_NEIGHBORS])

    return all_neighbors

def compute_drift_series(token, conn: psycopg.Connection):
    centroids = []
    slice_years = []
    slice_neighbors = []

    for start, end in SLICES:
        sid = f"{start}-{end}"
        centroid, vecs = compute_centroid(sid, token)

        if centroid is not None:
            centroids.append(centroid)
            slice_years.append(start)

            # Load FAISS index for this slice
            index = FaissIndex.load(str(faiss_slice_path((start, end))))
            neighbors = get_neighbors(conn, index, vecs)
            slice_neighbors.append(neighbors)

    if len(centroids) < 2:
        logger.warning(f"token='{token}' insufficient data for drift (n={len(centroids)})")
        return [], [], []

    drifts = []
    drift_x = []

    for i in range(1, len(centroids)):
        d = 1 - cosine(centroids[i], centroids[i - 1])
        drifts.append(d)
        drift_x.append(slice_years[i])

        logger.debug(f"token='{token}' drift {slice_years[i-1]}→{slice_years[i]} = {d:.4f}")

    logger.info(f"token='{token}' computed drift series (points={len(drifts)})")
    return drift_x, drifts, slice_neighbors

def main():
    logger.info("Starting drift + neighbor computation (canonical tokens only)")
    conn = get_connection()
    plt.figure(figsize=(12, 6))

    # terms = CONCEPT_SETS.keys()
    terms = ['liberty']

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing concept='{concept}' token='{token}'")

        x, y, neighbors = compute_drift_series(token, conn)

        if not y:
            logger.warning(f"Skipping concept='{concept}' (no drift data)")
            continue

        # Log neighbors for first slice as a test
        if neighbors and len(neighbors) > 0:
            for occ_neighbors in neighbors[0]:  # first slice
                for (tkn, can, doc, year), dist in occ_neighbors:
                    logger.info(f"Neighbor='{tkn}' (canonical='{can}'), doc={doc}, year={year}, sim={dist:.4f}")

        plt.plot(x, y, marker='o', label=concept)

    plt.xlabel("Year (start of slice)")
    plt.ylabel("Drift (1 - cosine)")
    plt.title("Per-slice Drift + Semantic Neighbors")
    plt.legend()
    plt.tight_layout()
    logger.info("Rendering plot")
    plt.show()
    logger.info("Done")

if __name__ == "__main__":
    main()
