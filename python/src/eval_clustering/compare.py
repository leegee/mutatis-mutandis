from .clustering_types import ClusterResult


def _graph_labels(result: ClusterResult):
    return result.communities


def compare_runs(density: ClusterResult, graph: ClusterResult):

    dmap = density.communities
    gmap = graph.communities

    return {
        "density_clusters": len(dmap),
        "graph_clusters": len(gmap),
    }
