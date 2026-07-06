from collections import defaultdict


def greedy_communities(graph):
    """
    Deterministic community extraction from kNN graph.

    Returns:
        dict[int, list[int]]
            cluster_id -> member node IDs

    Invariant:
        Output is always a mapping, never a flat label list.
    """

    if not graph:
        return {}

    # simple "connected-component" style greedy grouping
    visited = set()
    communities = defaultdict(list)

    cluster_id = 0

    def dfs(node, cid):
        stack = [node]

        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            communities[cid].append(n)

            for (nbr, _w) in graph.get(n, []):
                if nbr not in visited:
                    stack.append(nbr)

    for node in graph.keys():
        if node not in visited:
            dfs(node, cluster_id)
            cluster_id += 1

    return dict(communities)
