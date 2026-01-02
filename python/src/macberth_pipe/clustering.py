# src/macberth_pipe/clustering.py
from sklearn.cluster import KMeans
from .embedding import Embeddings
import numpy as np

def cluster_embeddings(emb: Embeddings, k: int = 20):
    """
    Cluster embeddings. k is reduced to the number of samples if necessary.
    Returns the cluster label for each row in emb.vectors.
    """
    X = emb.vectors
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)

    k_safe = min(int(k), n_samples)
    if k_safe <= 0:
        k_safe = 1

    km = KMeans(n_clusters=k_safe, random_state=42)
    labels = km.fit_predict(X)
    return labels
