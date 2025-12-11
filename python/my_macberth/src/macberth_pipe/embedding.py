# embeddings.py

from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
from pathlib import Path
import json
import hashlib

from .macberth_model import MacBERThModel
from .model_loader import get_local_macberth_path
from .types import Embeddings, ChunkMeta
from .embedding_store import EmbeddingsStore, save_embeddings, append_embeddings, load_embeddings


def stable_doc_id(text: str) -> str:
    """Generate a stable document ID by hashing the text content."""
    h = hashlib.sha1(text.encode('utf-8')).hexdigest()
    return f"doc_{h[:10]}"  # first 10 hex chars


def load_model(device: str = "cpu") -> MacBERThModel:
    model_path = get_local_macberth_path()
    return MacBERThModel(model_path=model_path, device=device)


def embed_documents(
    model: MacBERThModel,
    texts: List[str],
    device: str = "cpu",
    chunk_size: int = 512,
    average_chunks: bool = True,
    doc_meta: Optional[Dict[str, dict]] = None,
    store_dir: Optional[Path] = None,
    append_to_store: bool = False
) -> Embeddings:
    """
    Embed documents and optionally save or append to a disk store.
    """
    all_vecs = []
    all_ids = []
    metas = []

    for doc_i, text in enumerate(texts):
        doc_id = stable_doc_id(text)

        # Skip if doc_id already exists in store
        if store_dir and (store_dir / "ids.json").exists():
            try:
                with open(store_dir / "ids.json") as f:
                    existing_ids = json.load(f)
                if doc_id in existing_ids:
                    continue
            except Exception:
                pass

        meta_info = doc_meta.get(doc_id, {}) if doc_meta else {}

        # Corrected: embed_text returns a list; no extra () call
        chunk_vecs = model.embed_text(text, chunk_size=chunk_size)

        if not chunk_vecs:
            continue  # skip empty embedding results

        char_per_chunk = max(1, len(text) // len(chunk_vecs))
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

    if not all_vecs:
        return Embeddings(ids=[], vectors=np.empty((0, 0)), metas=[])

    vectors = np.vstack(all_vecs)
    emb = Embeddings(ids=all_ids, vectors=vectors, metas=metas)

    if store_dir:
        store = EmbeddingsStore(store_dir)
        if append_to_store:
            store.append(emb)
        else:
            store.save(emb)

    return emb


def load_embeddings_from_store(store_dir: Path) -> Embeddings:
    return load_embeddings(store_dir)
