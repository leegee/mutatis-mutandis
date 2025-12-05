from dataclasses import dataclass
import faiss
import numpy as np

@dataclass
class ChunkMeta:
    doc_id: str
    chunk_idx: int
    text: str
    start_char: int
    end_char: int
    title: str = ""
    author: str = ""
    year: str = ""
    permalink: str = ""

class SemanticIndex:
    def __init__(self, emb):
        self.emb = emb
        dim = emb.vectors.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(emb.vectors.astype("float32"))


def search(index: SemanticIndex, query_vec, top_k=5):
    q = np.asarray(query_vec, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)

    scores, idxs = index.index.search(q, top_k)

    results = []
    for rank, idx in enumerate(idxs[0]):
        meta = index.emb.metas[idx]
        results.append({
            "rank": rank,
            "score": float(scores[0, rank]),
            "doc_id": meta.doc_id,
            "chunk_idx": meta.chunk_idx,
            "text": meta.text,
            "start_char": meta.start_char,
            "end_char": meta.end_char
        })

    return results
