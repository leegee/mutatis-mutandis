from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class ChunkMeta:
    # TODO Finalise this list
    doc_id: str
    chunk_idx: int
    text: str
    start_char: int
    end_char: int
    title: str = ""
    author: str = ""
    year: str = ""
    permalink: str = ""

@dataclass(frozen=True)
class Embeddings:
    ids: List[str]
    vectors: np.ndarray
    metas: List[ChunkMeta]

@dataclass(frozen=True)
class QueryEmbeddings(Embeddings):
    """Embeddings where metadata is not relevant (e.g., user queries)."""

    @staticmethod
    def from_vectors(vectors: np.ndarray) -> "QueryEmbeddings":
        """
        Create a QueryEmbeddings with minimal dummy metadata.
        Enough for SemanticIndex & FAISS to function.
        """
        n = vectors.shape[0]

        dummy_metas = [
            ChunkMeta(
                doc_id=f"query_{i}",
                chunk_idx=i,
                text="",
                start_char=0,
                end_char=0
            ) for i in range(n)
        ]

        dummy_ids = [f"query_{i}" for i in range(n)]

        return QueryEmbeddings(
            ids=dummy_ids,
            vectors=vectors,
            metas=dummy_metas
        )
