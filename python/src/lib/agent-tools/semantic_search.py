"""
lib/agent_tools/semantic_search.py

Agent tool:
    Search the corpus by semantic similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from lib.macberth import MacBERThEmbedder


@dataclass
class SearchResult:
    doc_id: str
    score: float
    title: str | None = None


class SemanticSearchTool:
    """
    Agent-facing semantic search.

    The agent does not know about:
        - embeddings
        - FAISS
        - MacBERTh
        - tokenization

    It only knows:
        "search(query)"
    """

    def __init__(
        self,
        embedder: get_shared_embedder(),
        index,
        documents,
    ):
        self.embedder = embedder
        self.index = index
        self.documents = documents


    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> List[SearchResult]:

        vector = self.embedder.encode_normalized(
            query
        )

        vector = np.asarray(
            vector,
            dtype="float32",
        )

        scores, ids = self.index.search(
            vector.reshape(1, -1),
            limit,
        )

        results = []

        for score, idx in zip(
            scores[0],
            ids[0],
        ):

            doc = self.documents[idx]

            results.append(
                SearchResult(
                    doc_id=doc["doc_id"],
                    score=float(score),
                    title=doc.get("title"),
                )
            )

        return results
