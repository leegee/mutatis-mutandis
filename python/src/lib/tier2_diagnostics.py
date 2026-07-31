# lib/tier2_diagnostics.py

import numpy as np
from collections import Counter
from itertools import combinations

from lib.corpus_logging import logger

def knn_diagnostics(lookup, index, concept_forms, sample_n=25, k=25):
    """Dev utility: print kNN overlap and Jaccard stats for a concept's events."""
    forms     = {f.lower() for f in concept_forms}
    event_ids = list(lookup.iter_matching_event_ids(forms))

    if len(event_ids) < 5:
        print("Too few events")
        return

    event_ids = event_ids[:sample_n]
    vecs      = np.stack([lookup.get_event(eid)["embedding"] for eid in event_ids])
    _, nn_ids = index.search(vecs, k)
    knn_sets  = [set(map(int, row)) for row in nn_ids]

    overlaps  = []
    jaccards  = []
    entropies = []

    for i, j in combinations(range(len(knn_sets)), 2):
        a, b  = knn_sets[i], knn_sets[j]
        inter = len(a & b)
        union = len(a | b)
        overlaps.append(inter)
        jaccards.append(inter / union if union else 0)

    for s in knn_sets:
        flat    = list(s)
        freq    = Counter(flat)
        p       = np.array(list(freq.values())) / len(flat)
        entropy = -(p * np.log(p + 1e-9)).sum()
        entropies.append(entropy)

    print("\n--- KNN DIAGNOSTICS ---")
    print(f"events sampled: {len(event_ids)}")
    print(f"mean overlap: {np.mean(overlaps):.3f} ± {np.std(overlaps):.3f}")
    print(f"mean jaccard: {np.mean(jaccards):.3f} ± {np.std(jaccards):.3f}")
    print(f"mean entropy: {np.mean(entropies):.3f}")
    print("\noverlap quantiles:", np.percentile(overlaps, [0, 25, 50, 75, 100]))
    print("jaccard quantiles:", np.percentile(jaccards, [0, 25, 50, 75, 100]))

def audit_embedding_diversity(concept_name, query_vecs):
    logger.info("[tier2] EMBEDDING DIVERSITY AUDIT START")

    sample = query_vecs[:min(50, len(query_vecs))]
    if len(sample) < 2:
        return

    norms = np.linalg.norm(sample, axis=1)

    logger.info(
        f"[tier2] norms: mean={norms.mean():.6f} std={norms.std():.6f} "
        f"min={norms.min():.6f} max={norms.max():.6f}"
    )

    normed = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T

    n = len(sample)
    off = sim[~np.eye(n, dtype=bool)]

    logger.info(
        f"[tier2] cosine: mean={off.mean():.6f} std={off.std():.6f} "
        f"p95={np.percentile(off, 95):.6f} max={off.max():.6f}"
    )


def audit_embedding_isotropy(vecs):
    logger.info("[tier2] ISOTROPY AUDIT START")

    v = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    cov = np.cov(v.T)

    eigvals = np.linalg.eigvalsh(cov)[::-1]
    ratio = eigvals / (eigvals.sum() + 1e-12)

    logger.info(
        f"[tier2] eig_top1={eigvals[0]:.6f} "
        f"explained_top1={ratio[0]:.4f} "
        f"explained_top5={ratio[:5].sum():.4f}"
    )


def audit_hubness(index, vecs, k=25):
    logger.info("[tier2] HUBNESS AUDIT START")

    _, nn = index.search(vecs, k)
    flat = nn.flatten()

    freq = Counter(flat)
    vals = np.array(list(freq.values()))

    logger.info(
        f"[tier2] hubness mean={vals.mean():.3f} "
        f"std={vals.std():.3f} max={vals.max():.3f}"
    )


def audit_neighbour_identity(all_neigh_ids):
    flat = all_neigh_ids.flatten()
    if not len(flat):
        return

    freq = Counter(flat)

    logger.info("[tier2] TOP NEIGHBOUR IDS")
    for k, v in freq.most_common(10):
        logger.info(f"[tier2] id={k} freq={v}")


def audit_knn_stability(index, lookup, event_ids, k=25):
    logger.info("[tier2] KNN STABILITY AUDIT START")

    vecs = np.stack([lookup.get_event(e)["embedding"] for e in event_ids])
    _, nn = index.search(vecs, k)

    scores = []

    for i in range(len(nn) - 1):
        a = set(nn[i])
        b = set(nn[i + 1])
        scores.append(len(a & b) / (len(a | b) + 1e-12))

    logger.info(
        f"[tier2] stability mean={np.mean(scores):.4f} std={np.std(scores):.4f}"
    )
