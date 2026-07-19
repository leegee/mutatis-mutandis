import numpy as np


def build_knn_graph(substrate, event_ids, k=25):
    """
    FAISS kNN graph over Zarr-derived embeddings.

    substrate provides:
        - substrate.index (FAISS)
        - substrate.lookup (ZarrEventLookup)
    """

    if len(event_ids) == 0:
        return {}

    # embedding matrix from Zarr is the canonical truth
    X = np.vstack([
        substrate.lookup.get_ensemble_embedding(substrate.lookup._pos[int(eid)])
        for eid in event_ids
    ]).astype(np.float32)

    distances, neighbors = substrate.index.search(X, k)

    graph = {}

    for i, src_id in enumerate(event_ids):
        edges = []
        for j in range(k):
            tgt = int(neighbors[i, j])
            if tgt == -1:
                continue
            edges.append((tgt, float(distances[i, j])))
        graph[int(src_id)] = edges

    return graph
