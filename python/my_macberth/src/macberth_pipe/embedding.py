from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from macberth_pipe.semantic import ChunkMeta

@dataclass(frozen=True)
class Embeddings:
    ids: List[str]          # doc_id or doc_id_chunkN
    vectors: np.ndarray     # (n, dim)
    metas: List[ChunkMeta]  # metadata per vector

def load_model(name="emanjavacas/MacBERTh"):
    """
    Load MacBERTh tokenizer and model
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    return tokenizer, model

def embed_documents(
    tokenizer, 
    model, 
    texts: List[str], 
    device="cpu", 
    chunk_size=512,
    average_chunks=True,
    doc_meta: Optional[Dict[str, dict]] = None
) -> Embeddings:
    """
    Embed documents with optional chunk averaging and snippet-level metadata.
    """
    all_vecs = []
    all_ids = []
    metas: List[ChunkMeta] = []

    for doc_i, text in enumerate(texts):
        doc_id = f"doc{doc_i}"
        meta_info = doc_meta.get(doc_id, {}) if doc_meta else {}

        # Tokenize without truncation
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"][0]

        # Split into chunks
        chunks = [input_ids[j:j+chunk_size] for j in range(0, len(input_ids), chunk_size)]

        # Approximate token→char alignment
        tokens = tokenizer.tokenize(text)
        char_per_token = max(1, len(text) // max(1, len(tokens)))

        chunk_vecs = []
        chunk_metas = []

        for ci, chunk in enumerate(chunks):
            chunk_enc = {"input_ids": chunk.unsqueeze(0).to(device)}
            with torch.no_grad():
                out = model(**chunk_enc)
            vec = out.last_hidden_state.mean(dim=1).cpu().numpy()[0].astype(np.float32)
            chunk_vecs.append(vec)

            start_char = ci * chunk_size * char_per_token
            end_char = min(len(text), (ci+1) * chunk_size * char_per_token)
            chunk_text = text[start_char:end_char]

            chunk_metas.append(
                ChunkMeta(
                    doc_id=doc_id,
                    chunk_idx=ci,
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    title=meta_info.get("title",""),
                    author=meta_info.get("author",""),
                    year=meta_info.get("year",""),
                    permalink=meta_info.get("permalink","")
                )
            )

        chunk_vecs = np.vstack(chunk_vecs)

        if average_chunks:
            all_vecs.append(chunk_vecs.mean(axis=0, keepdims=True))
            all_ids.append(doc_id)
            metas.append(chunk_metas[0])  # representative
        else:
            all_vecs.append(chunk_vecs)
            all_ids.extend([f"{doc_id}_chunk{ci}" for ci in range(len(chunk_vecs))])
            metas.extend(chunk_metas)

    vectors = np.vstack(all_vecs)
    return Embeddings(ids=all_ids, vectors=vectors, metas=metas)
