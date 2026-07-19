from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compare_labels(a, b):
    """
    Standard clustering agreement metrics.
    """

    return {
        "ARI": float(adjusted_rand_score(a, b)),
        "NMI": float(normalized_mutual_info_score(a, b))
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def cluster_stability(cluster_sets_a, cluster_sets_b):
    """
    Measures how stable clusters are across two systems or time slices.
    """

    scores = []

    for ca in cluster_sets_a:
        best = max(jaccard(ca, cb) for cb in cluster_sets_b)
        scores.append(best)

    return sum(scores) / max(1, len(scores))
