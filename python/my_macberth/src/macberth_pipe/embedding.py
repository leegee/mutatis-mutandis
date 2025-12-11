from dataclasses import dataclass
from typing import List, Optional, Dict, Union
import numpy as np
from pathlib import Path
import hashlib
import logging

from .macberth_model import MacBERThModel
from .model_loader import get_local_macberth_path
from .types import Embeddings, ChunkMeta
from .faiss_store import FaissStore

logger = logging.getLogger(__name__)


def stable_doc_id(text: str) -> str:
    """Generate a stable document ID by hashing the text content."""
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"doc_{h[:10]}"

def load_model(device: str = "cpu") -> MacBERThModel:
    model_path = get_local_macberth_path()
    return MacBERThModel(model_path=model_path, device=device)


def embed_chunks_batched(model: MacBERThModel, chunks: List[str], batch_size: int) -> List[np.ndarray]:
    """Embed a list of chunk strings using batched inference."""
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.debug(f"Embedding batch of size {len(batch)}")
        batch_vecs = model.embed_text(batch)  # now supports list input
        all_vecs.extend(batch_vecs)
    return all_vecs

def embed_documents(
    model: MacBERThModel,
    texts: List[str],
    device: str = "cpu",
    chunk_size: int = 512,
    average_chunks: bool = False,
    doc_meta: Optional[Dict[str, dict]] = None,
    store_dir: Optional[Path] = None,
    sqlite_db: Optional[Path] = None,
    append_to_store: bool = False,
    batch_size: int = 8
) -> Embeddings:

    all_vecs = []
    all_ids = []
    metas = []

    for text in texts:
        doc_id = stable_doc_id(text)
        logger.debug(f"Processing document {doc_id}")

        meta_info = doc_meta.get(doc_id, {}) if doc_meta else {}
        chunks = model.split_into_chunks(text, chunk_size=chunk_size)
        if not chunks:
            continue

        chunk_vecs = embed_chunks_batched(model, chunks, batch_size)

        char_per_chunk = max(1, len(text) // len(chunk_vecs))
        chunk_metas = []
        for ci, vec in enumerate(chunk_vecs):
            start_char = ci * char_per_chunk
            end_char   = min(len(text), (ci + 1) * char_per_chunk)
            chunk_text = text[start_char:end_char]

            chunk_metas.append(ChunkMeta(
                doc_id=doc_id,
                chunk_idx=ci,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                title=meta_info.get("title", ""),
                author=meta_info.get("author", ""),
                year=meta_info.get("year", ""),
                permalink=meta_info.get("permalink", "")
            ))

        chunk_vecs = np.vstack(chunk_vecs)

        if average_chunks:
            all_vecs.append(chunk_vecs.mean(axis=0, keepdims=True))
            all_ids.append(doc_id)
            metas.append(chunk_metas[0])
        else:
            all_vecs.append(chunk_vecs)
            all_ids.extend([f"{doc_id}_chunk{ci}" for ci in range(len(chunk_vecs))])
            metas.extend(chunk_metas)

    if not all_vecs:
        return Embeddings(ids=[], vectors=np.empty((0, 0)), metas=[])

    vectors = np.vstack(all_vecs)
    emb = Embeddings(ids=all_ids, vectors=vectors, metas=metas)

    # Save embeddings and persist to FAISS + SQLite if requested
    if store_dir:
        store_dir.mkdir(parents=True, exist_ok=True)
        faiss_store = FaissStore(store_dir, sqlite_db=sqlite_db)
        if append_to_store:
            faiss_store.append(emb)
        else:
            faiss_store.build(emb)
        logger.debug(f"Saved {len(all_ids)} embeddings to FAISS store at {store_dir}")

    return emb
