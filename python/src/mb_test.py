#!/usr/bin/env python

import json
import time
import numpy as np
from collections import Counter
import hdbscan

from mb_embedding_pipeline import load_vectors
from lib.js_divergence import js_divergence
from lib.eebo_logging import logger
from lib.FaissIndex import FaissIndex
from lib.mb_paths import faiss_slice_path
from lib.eebo_db import get_connection
from lib.eebo_config import OUT_DIR, CONCEPT_SETS
from lib.wordlist import STOPWORDS
from lib.eebo_config import SLICES

MIN_TOKENS = 30

OUT_PATH = OUT_DIR / "drift_neighbors_micro_senses_slices.json"


K_NEIGHBORS = 5
TOP_K_NEIGHBORS = 15

_FAISS_CACHE = {}
_CACHE_FILE = OUT_DIR / "token_occurrence_cache.json"
_ID_CACHE = {}


def get_faiss_index(slice_range):
    if slice_range in _FAISS_CACHE:
        logger.debug(f"[{slice_range}] FAISS cache hit")
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


def detect_phase_transitions(slices_data, min_tokens=30, minor_cluster_threshold=10, single_doc_ratio=0.5):
    """
    Detects both major and minor semantic phase transitions in slices_data.

    Args:
        slices_data: list of dicts, each with keys:
            'year', 'cluster_sizes', 'drift', 'js_divergence', 'births', 'deaths', 'top_docs'
        min_tokens: minimum tokens per slice to consider reliable
        minor_cluster_threshold: cluster size below which a cluster may trigger a minor shift
        single_doc_ratio: fraction of top-doc tokens that marks a potential single-doc spike

    Returns:
        dict with keys:
            'major': list of major transitions
            'minor': list of minor / subtle shifts
            'single_doc_spikes': list of potential artefacts
    """
    if len(slices_data) < 3:
        return {"major": [], "minor": [], "single_doc_spikes": []}

    token_counts = np.array([sum(s["cluster_sizes"]) if s["cluster_sizes"] else 0 for s in slices_data])
    drifts = np.array([s["drift"] for s in slices_data[1:]])
    jsd = np.array([s["js_divergence"] for s in slices_data[1:]])
    births = np.array([s["births"] for s in slices_data[1:]])
    # deaths = np.array([s["deaths"] for s in slices_data[1:]])

    valid = token_counts[1:] >= min_tokens

    def zscore(x):
        if np.std(x) == 0:
            return np.zeros_like(x)
        return (x - np.mean(x)) / np.std(x)

    drift_z = zscore(drifts)
    jsd_z = zscore(jsd)
    births_z = zscore(births)
    # deaths_z = zscore(deaths)

    score = drift_z + jsd_z + 0.5 * births_z

    major, minor, single_doc_spikes = [], [], []

    for i, s in enumerate(score):
        if not valid[i]:
            continue

        slice_data = slices_data[i + 1]

        # Detect major transitions
        if s > 1.5 and drift_z[i] > 0.5 and jsd_z[i] > 0.5:
            major.append({
                "year": slice_data["year"],
                "score": float(s),
                "drift": slice_data["drift"],
                "js_divergence": slice_data["js_divergence"],
                "births": slice_data["births"],
                "deaths": slice_data["deaths"],
                "count": int(token_counts[i + 1])
            })

        # Detect minor transitions: small cluster birth/death
        if slice_data["births"] > 0 or slice_data["deaths"] > 0:
            small_clusters = [c for c in slice_data["cluster_sizes"] if c <= minor_cluster_threshold]
            if small_clusters:
                minor.append({
                    "year": slice_data["year"],
                    "score": float(s),
                    "small_cluster_count": len(small_clusters),
                    "births": slice_data["births"],
                    "deaths": slice_data["deaths"],
                    "count": int(token_counts[i + 1])
                })

        # Detect single-doc spikes
        if "top_docs" in slice_data and slice_data["top_docs"]:
            top_doc_count = slice_data["top_docs"][0][1]  # most common doc
            if top_doc_count / token_counts[i + 1] > single_doc_ratio:
                single_doc_spikes.append({
                    "year": slice_data["year"],
                    "top_doc": slice_data["top_docs"][0][0],
                    "top_doc_count": top_doc_count,
                    "cluster_size": slice_data["cluster_sizes"][0] if slice_data["cluster_sizes"] else 0,
                    "count": int(token_counts[i + 1])
                })

    return {
        "major": major,
        "minor": minor,
        "single_doc_spikes": single_doc_spikes
    }


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
        neighbor_sims = []
        doc_ids = []

        for i, vec in enumerate(vecs):
            distances, neighbor_ids = index.search(vec.reshape(1, -1), K_NEIGHBORS * 5)
            rows = lookup_token_occurrences(conn, neighbor_ids[0].tolist())

            if i < 10:
                doc_ids.extend([doc for (_, _, doc, _) in rows])

            for (tkn, _can, _doc, _year), sim in zip(rows, distances[0], strict=True):
                if sim < 0.6 or tkn == token:
                    continue
                neighbor_tokens.append(tkn)
                neighbor_sims.append(sim)

        # --- FILTER STOPWORDS ---
        filtered = [
            (t, s)
            for t, s in zip(neighbor_tokens, neighbor_sims)
            if t.lower() not in STOPWORDS
        ]

        # --- DEDUPLICATE + AGGREGATE ---
        from collections import defaultdict

        neighbor_map = defaultdict(lambda: {"count": 0, "sim_sum": 0.0})

        for t, s in filtered:
            neighbor_map[t]["count"] += 1
            neighbor_map[t]["sim_sum"] += s

        # --- BUILD DISTRIBUTION (for JSD) ---
        curr_counts = Counter({t: v["count"] for t, v in neighbor_map.items()})

        # --- BUILD TOP NEIGHBORS ---
        top_neighbors = [
            {
                "token": t,
                "count": v["count"],
                "similarity": float(v["sim_sum"] / v["count"])
            }
            for t, v in neighbor_map.items()
        ]

        # sort by similarity (semantic proximity)
        top_neighbors.sort(key=lambda x: -x["similarity"])
        top_neighbors = top_neighbors[:TOP_K_NEIGHBORS]

        # --- ENTROPY ---
        ent = entropy_from_tokens([t for t, _ in filtered])

        logger.info(
            f"[{sid}] token='{token}' "
            f"clusters={len(cluster_centroids)} sizes={cluster_sizes} "
            f"entropy={ent:.4f} neighbors={len(filtered)} unique={len(neighbor_map)}"
        )

        if doc_ids:
            logger.info(
                f"[{sid}] top_docs="
                + ", ".join(f"{d}:{c}" for d, c in Counter(doc_ids).most_common(5))
            )

        logger.info(
            f"[{sid}] top_neighbors="
            + ", ".join(
                f"{n['token']}:{n['count']}"
                for n in top_neighbors[:8]
            )
        )

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

        # --- DRIFT + JSD ---
        if prev_clusters:
            matches, deaths = match_clusters(prev_clusters, cluster_centroids)

            movements = [d for p, c, d in matches if p is not None]
            drift_val = float(np.mean(movements)) if movements else 1.0
            births = sum(1 for p, c, d in matches if p is None)

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
        prev_clusters = cluster_centroids
        prev_year = start

        slices_data.append(slice_entry)

    return slices_data


def log_phase_transitions(token, transitions, logger):
    """
    Log detected phase transitions for a token in a structured timeline format.

    Parameters:
    - token (str): the token being analyzed
    - transitions (dict): output from detect_phase_transitions()
    - logger: logging.Logger instance
    """
    if not transitions:
        logger.info(f"[{token}] No phase transitions detected.")
        return

    # Major transitions
    for t in transitions.get("major", []):
        logger.info(
            f"[{token}] MAJOR PHASE SHIFT @ {t['year']} "
            f"(score={t['score']:.2f}, drift={t['drift']:.4f}, "
            f"jsd={t['js_divergence']:.4f}, births={t['births']}, deaths={t['deaths']})"
        )

    # Minor transitions
    for t in transitions.get("minor", []):
        logger.info(
            f"[{token}] MINOR SHIFT @ {t['year']} "
            f"(small_clusters={t['small_cluster_count']}, "
            f"births={t['births']}, deaths={t['deaths']})"
        )

    # Single-doc spikes
    for t in transitions.get("single_doc_spikes", []):
        logger.info(
            f"[{token}] SINGLE-DOC SPIKE @ {t['year']} "
            f"(top_doc={t['top_doc']}, count={t['top_doc_count']}, "
            f"cluster_size={t['cluster_size']})"
        )


def main():
    start = time.time()
    logger.info("Starting cluster-aware drift + phase transition detection")

    load_id_cache()
    conn = get_connection()

    terms = CONCEPT_SETS.keys() # ['liberty', 'freedom']
    results = {}

    for concept in terms:
        token = concept.lower()
        logger.info(f"Processing '{token}'")

        slices_data = compute_drift_and_neighbors_clustered(token, conn)

        if not slices_data:
            continue

        transitions = detect_phase_transitions(slices_data)

        log_phase_transitions(token, transitions, logger)

        results[token] = {
            "slices": slices_data,
            "phase_transitions": transitions
        }

    save_id_cache()

    logger.info(f"Elapsed time: {time.time() - start:.3f} seconds")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved dataset to {OUT_PATH}")


if __name__ == "__main__":
    main()
