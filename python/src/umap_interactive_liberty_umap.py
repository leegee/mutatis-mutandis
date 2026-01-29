#!/usr/bin/env python
"""
umap_interactive_liberty_umap.py

Interactive UMAP projection of all LIBERTY-context vectors across slices.
Mouse-over shows the token.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pandas as pd
import faiss
import umap
import plotly.express as px

import lib.eebo_config as config
from lib.faiss_slices import faiss_slice_path, vocab_slice_path
from lib.wordlist import STOPWORDS
from lib.eebo_logging import logger

WINDOW = 5
LIBERTY_FORMS = config.CONCEPT_SETS["LIBERTY"]["forms"]
FALSE_POSITIVES = config.CONCEPT_SETS["LIBERTY"]["false_positives"]



def get_liberty_context_vectors(index, vocab: List[str]) -> Tuple[List[str], np.ndarray, List[int]]:
    """
    Returns:
        - list of context tokens
        - stacked vectors (dense np.ndarray)
        - slice index for each token
    """
    token_to_idx = {token.lower(): i for i, token in enumerate(vocab)}
    context_tokens: List[str] = []
    vectors_list: List[np.ndarray] = []
    slice_ids: List[int] = []

    for form in LIBERTY_FORMS:
        idx = token_to_idx.get(form.lower())
        if idx is None:
            continue

        start = max(idx - WINDOW, 0)
        end = min(idx + WINDOW + 1, len(vocab))

        for i in range(start, end):
            token = vocab[i].lower()
            if token in STOPWORDS or token in FALSE_POSITIVES:
                continue

            # no canonicalisation here: keep original token
            vec = index.reconstruct(i)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

            vectors_list.append(vec)
            context_tokens.append(token)
            slice_ids.append(0)  # placeholder; will update per slice

    if not vectors_list:
        raise RuntimeError("No LIBERTY-context tokens found in slice")

    return context_tokens, np.stack(vectors_list).astype(np.float32), slice_ids



all_tokens: List[str] = []
all_vectors: List[np.ndarray] = []
all_slices: List[int] = []

for slice_idx, slice_range in enumerate(config.SLICES):
    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)

    if not index_path.exists() or not vocab_path.exists():
        continue

    logger.debug(f"Processing slice {slice_range[0]}–{slice_range[1]}")

    index = faiss.read_index(str(index_path))
    vocab = [line.strip() for line in open(vocab_path, encoding="utf-8")]

    tokens, vectors, slice_ids = get_liberty_context_vectors(index, vocab)
    if len(tokens) < 5:
        continue

    # update slice_ids
    slice_ids = [slice_idx] * len(tokens)

    all_tokens.extend(tokens)
    all_vectors.append(vectors)
    all_slices.extend(slice_ids)

# flatten vectors
all_vectors_np = np.vstack(all_vectors)  # shape: (total_tokens, vector_dim)
logger.info(f"Total tokens for UMAP: {all_vectors_np.shape[0]}")


# UMAP projection
reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
embedding = reducer.fit_transform(all_vectors_np)
embedding = np.array(embedding)  # ensure dense for slicing

x = embedding[:, 0]
y = embedding[:, 1]


# interactive plot
df_plot = pd.DataFrame({
    "x": x,
    "y": y,
    "token": all_tokens,
    "slice_idx": all_slices,
    "slice_label": [f"{config.SLICES[i][0]}–{config.SLICES[i][1]}" for i in all_slices]
})

fig = px.scatter(
    df_plot,
    x="x",
    y="y",
    color="slice_label",
    hover_data=["token", "slice_label"],
    title="UMAP of LIBERTY-context tokens across slices"
)

fig.update_traces(
    marker=dict(size=6, opacity=0.7),
        hoverlabel=dict(
        bgcolor="black",
        font_size=12,
        font_color="white"
    )
)

fig.update_layout(width=1200, height=800)
fig.write_html("umap_liberty_interactive.html")
logger.info("Wrote umap_liberty_interactive.html")
