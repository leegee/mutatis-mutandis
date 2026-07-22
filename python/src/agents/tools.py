def semantic_search(query: str, limit: int = 10):
    """
    Find documents semantically related to a query.
    """

    embeddings = model.encode([query])

    distances, ids = index.search(
        embeddings,
        limit
    )

    results = []

    for doc_id, score in zip(ids[0], distances[0]):
        results.append({
            "doc_id": doc_id,
            "score": float(score),
            "title": get_title(doc_id)
        })

    return results


def get_document(doc_id):
    return {
        "doc_id": doc_id,
        "title": "...",
        "year": 1645,
        "author": "...",
        "text": "..."
    }


def cluster_summary(cluster_id):
    # ..
    return {
        "cluster": 42,
        "label": "Divine Providence",
        "size": 342,
        "period": "1640-1660",
        "top_terms": [
            "God",
            "judgement",
            "covenant"
        ]
    }


def find_neighbours(doc_id, k=10):
    # ...
    return [
       {
        "doc_id": "...",
        "similarity": 0.91
       }
    ]

