#!/usr/bin/env python
import psycopg
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

MODEL_PATH = "/s/src/pamphlets/python/lib/macberth-huggingface"
BATCH_SIZE = 32
TOKEN_WINDOW = 5  # ±5 tokens if sentence unavailable

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH).to(device)
model.eval()

conn = psycopg.connect("")
conn.autocommit = True


def embed_texts(texts, batch_size=BATCH_SIZE, max_length=128):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            # mean pooling over token embeddings
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings)


def cluster_embeddings(embeddings, n_clusters=None, distance_threshold=1.0):
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=None if n_clusters else distance_threshold,
        affinity="cosine",
        linkage="average"
    )
    return clustering.fit_predict(embeddings)


def build_token_windows(rows, window_size=TOKEN_WINDOW):
    """
    Construct ±window_size token windows for tokens without sentence context.
    rows: list of (doc_id, token_idx, token, sentence_text_norm)
    Returns: list of dicts with keys doc_id, token_idx, surface_form, text
    """
    contexts = []
    # Group tokens by document
    from collections import defaultdict
    doc_tokens = defaultdict(list)
    for doc_id, token_idx, token, sentence in rows:
        doc_tokens[doc_id].append((token_idx, token, sentence))

    for doc_id, tokens in doc_tokens.items():
        # Sort by token_idx
        tokens.sort(key=lambda x: x[0])
        for i, (token_idx, token, sentence) in enumerate(tokens):
            if sentence:
                text = sentence
            else:
                # Build ±window around token
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                text = " ".join(t[1] for t in tokens[start:end])
            contexts.append({
                "doc_id": doc_id,
                "token_idx": token_idx,
                "surface_form": token,
                "text": text
            })
    return contexts


def canonicalize_tokens():
    with conn.cursor() as cur:
        # Extract all tokens with sentence context
        cur.execute("""
            SELECT t.doc_id, t.token_idx, t.token,
                   s.sentence_text_norm
            FROM tokens t
            LEFT JOIN sentences s
              ON t.doc_id = s.doc_id
             AND t.sentence_id = s.sentence_id
            ORDER BY t.doc_id, t.token_idx
        """)
        rows = cur.fetchall()

    # Build sentence-first, token-window fallback contexts
    print(f"[INFO] Building contexts for {len(rows)} tokens...")
    contexts = build_token_windows(rows)

    # Embed contexts
    texts = [c["text"] for c in contexts]
    print(f"[INFO] Embedding {len(texts)} contexts...")
    embeddings = embed_texts(texts)

    # Cluster embeddings
    print("[INFO] Clustering embeddings...")
    labels = cluster_embeddings(embeddings)

    # Assign canonical = most frequent surface form per cluster
    clusters = {}
    for ctx, label in zip(contexts, labels, strict=True):
        clusters.setdefault(label, []).append(ctx["surface_form"])
    canonical_map = {label: max(set(forms), key=forms.count) for label, forms in clusters.items()}

    for ctx, label in zip(contexts, labels, strict=True):
        ctx["canonical"] = canonical_map[label]

    # Update Postgres tokens table
    print("[INFO] Writing canonical forms back to tokens table...")
    with conn.cursor() as cur:
        for ctx in tqdm(contexts):
            cur.execute("""
                UPDATE tokens
                SET canonical = %s
                WHERE doc_id = %s AND token_idx = %s
            """, (ctx["canonical"], ctx["doc_id"], ctx["token_idx"]))

    print("[DONE] Canonicalization complete.")


if __name__ == "__main__":
    canonicalize_tokens()
