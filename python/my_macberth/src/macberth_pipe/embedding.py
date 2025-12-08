# python/my_macberth/src/macberth_pipe/embedding.py

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from .macberth_model import MacBERThModel
from .model_loader import get_local_macberth_path
from .semantic import ChunkMeta


@dataclass(frozen=True)
class Embeddings:
    ids: List[str]
    vectors: np.ndarray
    metas: List[ChunkMeta]


def load_model(device="cpu"):
    """
    Unified model loader – uses shared logic from model_loader.py.
    No hardcoded paths.
    """
    model_path = get_local_macberth_path()
    return MacBERThModel(model_path=model_path, device=device)


def embed_documents(
    model: MacBERThModel,
    texts: List[str],
    device="cpu",
    chunk_size=512,
    average_chunks=True,
    doc_meta: Optional[Dict[str, dict]] = None
) -> Embeddings:
    """
    Embed a list of documents with optional averaging and chunk metadata.
    """
    all_vecs = []
    all_ids = []
    metas = []

    for doc_i, text in enumerate(texts):
        doc_id = f"doc{doc_i}"
        meta_info = doc_meta.get(doc_id, {}) if doc_meta else {}

        chunk_vecs = model.embed_text(text, chunk_size=chunk_size)

        # Approximate char alignment
        char_per_chunk = max(1, len(text) // max(1, len(chunk_vecs)))
        chunk_metas = []
        for ci, vec in enumerate(chunk_vecs):
            start_char = ci * char_per_chunk
            end_char = min(len(text), (ci + 1) * char_per_chunk)
            chunk_text = text[start_char:end_char]
            chunk_metas.append(
                ChunkMeta(
                    doc_id=doc_id,
                    chunk_idx=ci,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    title=meta_info.get("title", ""),
                    author=meta_info.get("author", ""),
                    year=meta_info.get("year", ""),
                    permalink=meta_info.get("permalink", "")
                )
            )

        chunk_vecs = np.vstack(chunk_vecs)

        if average_chunks:
            all_vecs.append(chunk_vecs.mean(axis=0, keepdims=True))
            all_ids.append(doc_id)
            metas.append(chunk_metas[0])
        else:
            all_vecs.append(chunk_vecs)
            all_ids.extend([f"{doc_id}_chunk{ci}" for ci in range(len(chunk_vecs))])
            metas.extend(chunk_metas)

    vectors = np.vstack(all_vecs)
    return Embeddings(ids=all_ids, vectors=vectors, metas=metas)
