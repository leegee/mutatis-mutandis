import numpy as np
from typing import Dict

def compute_drift(word: str, period_embeddings: Dict[str, np.ndarray]):
    periods = sorted(period_embeddings.keys())
    centroids = {p: period_embeddings[p].mean(axis=0) for p in periods}

    drift = {}
    for a, b in zip(periods[:-1], periods[1:]):
        v1, v2 = centroids[a], centroids[b]
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        drift[(a,b)] = 1 - sim
    return drift
