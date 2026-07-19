from .graph_builder import build_knn_graph
from .communities import greedy_communities
from .clustering_types import ClusterResult


def run_graph_pipeline(substrate, event_ids, k: int = 25) -> ClusterResult:

    graph = build_knn_graph(
        substrate,
        event_ids,
        k=k
    )

    communities = greedy_communities(graph)

    membership: dict[int, int] = {}

    for cid, nodes in communities.items():
        for nid in nodes:
            membership[int(nid)] = int(cid)

    labels = [membership.get(int(eid), -1) for eid in event_ids]

    return ClusterResult(
        event_ids=list(event_ids),
        membership=membership,
        communities=communities,
        graph=graph,
        labels=labels
    )
