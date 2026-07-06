import numpy as np
import hdbscan

from .clustering_types import ClusterResult


def run_density_pipeline(substrate, event_ids) -> ClusterResult:

    X = np.vstack([
        substrate.lookup.get_ensemble_embedding(substrate.lookup.get_pos(eid))
        for eid in event_ids
    ])

    labels = hdbscan.HDBSCAN(
        metric="euclidean",   # IMPORTANT: no cosine ambiguity
        min_cluster_size=10
    ).fit_predict(X)

    membership = {
        int(eid): int(cid)
        for eid, cid in zip(event_ids, labels)
    }

    communities: dict[int, list[int]] = {}
    for eid, cid in membership.items():
        communities.setdefault(cid, []).append(eid)

    return ClusterResult(
        event_ids=list(event_ids),
        membership=membership,
        communities=communities,
        graph=None,
        labels=list(labels)
    )
