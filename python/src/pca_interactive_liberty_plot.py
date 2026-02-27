#!/usr/bin/env python
"""
pca_interactive_liberty_plot.py

Interactive plot of LIBERTY-conditioned conceptual poles across slices.

Supports:
- Raw FAISS slice vectors
- Reference-aligned vectors (Orthogonal Procrustes)

Geometry choice is handled via dispatch, not Optionals.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import faiss
from sklearn.decomposition import PCA
import pandas as pd
from pathlib import Path
from collections import Counter

import plotly.express as px

import lib.eebo_config as config
from lib.faiss_slices import faiss_slice_path, vocab_slice_path
from lib.eebo_logging import logger
from lib.wordlist import STOPWORDS
from align import load_aligned_vectors


USE_ALIGNED_VECTORS = True

TOP_N_WORDS = 15
WINDOW = 5
LIBERTY_FORMS = config.CONCEPT_SETS["LIBERTY"]["forms"]
FALSE_POSITIVES = config.CONCEPT_SETS["LIBERTY"]["false_positives"]


# Vector Extraction
def get_vectors_from_aligned(
    slice_id: str,
    vocab: List[str],
) -> Tuple[List[str], np.ndarray, Counter[str]]:

    aligned_vectors = load_aligned_vectors(slice_id)

    token_to_idx = {token.lower(): i for i, token in enumerate(vocab)}
    context_tokens: List[str] = []
    vectors: List[np.ndarray] = []
    counts: Counter[str] = Counter()

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

            vec = aligned_vectors.get(token)
            if vec is None:
                continue

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            vectors.append(vec)
            context_tokens.append(token)
            counts[token] += 1

    if not vectors:
        raise RuntimeError(f"No LIBERTY-context tokens found in slice {slice_id}")

    return context_tokens, np.stack(vectors).astype(np.float32), counts


def get_vectors_from_faiss(
    slice_range: Tuple[int, int],
    vocab: List[str],
) -> Tuple[List[str], np.ndarray, Counter[str]]:

    index_path = faiss_slice_path(slice_range)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found for slice {slice_range}")

    index = faiss.read_index(str(index_path))

    token_to_idx = {token.lower(): i for i, token in enumerate(vocab)}
    context_tokens: List[str] = []
    vectors: List[np.ndarray] = []
    counts: Counter[str] = Counter()

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

            vec = index.reconstruct(i)

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            vectors.append(vec)
            context_tokens.append(token)
            counts[token] += 1

    if not vectors:
        raise RuntimeError(f"No LIBERTY-context tokens found in slice {slice_range}")

    return context_tokens, np.stack(vectors).astype(np.float32), counts


# PCA Pole Computation
def compute_poles(
    word_vectors: np.ndarray,
    words: List[str],
    top_n: int = TOP_N_WORDS,
):

    centroid = word_vectors.mean(axis=0)
    X = word_vectors - centroid

    pca = PCA(n_components=1)
    pca.fit(X)

    pc1 = pca.components_[0]
    pc1 /= np.linalg.norm(pc1)

    scores = X @ pc1

    top_idx = np.argsort(scores)[-top_n:][::-1]
    bottom_idx = np.argsort(scores)[:top_n]

    return (
        [words[i] for i in top_idx],
        [words[i] for i in bottom_idx],
        scores,
        float(pca.explained_variance_ratio_[0]),
    )


# Main
all_words: List[str] = []
all_slices: List[int] = []
all_scores: List[float] = []
all_poles: List[str] = []
all_counts: List[int] = []

for slice_range in config.SLICES:

    slice_id = f"{slice_range[0]}-{slice_range[1]}"
    vocab_path = vocab_slice_path(slice_range)

    if not vocab_path.exists():
        continue

    logger.debug(f"=== Slice {slice_id} ===")

    vocab = [line.strip() for line in open(vocab_path, encoding="utf-8")]

    if USE_ALIGNED_VECTORS:
        words, vectors, counts = get_vectors_from_aligned(slice_id, vocab)
    else:
        words, vectors, counts = get_vectors_from_faiss(slice_range, vocab)

    if len(words) < 10:
        continue

    pos_words, neg_words, scores, var = compute_poles(vectors, words)

    for i, word in enumerate(words):
        pole_type = (
            "positive" if word in pos_words
            else "negative" if word in neg_words
            else "other"
        )

        all_words.append(word)
        all_slices.append(slice_range[0])
        all_scores.append(float(scores[i]))
        all_poles.append(pole_type)
        all_counts.append(counts[word])


df = pd.DataFrame({
    "word": all_words,
    "slice_start": all_slices,
    "score": all_scores,
    "pole": all_poles,
    "count": all_counts,
})


fig = px.scatter(
    df,
    x="slice_start",
    y="score",
    color="pole",
    hover_data=["word", "count", "slice_start", "pole"],
    color_discrete_map={
        "positive": "cyan",
        "negative": "orange",
        "other": "lightgray",
    },
    labels={
        "slice_start": "Slice Start Year",
        "score": "PC1 Projection",
    },
)

fig.update_layout(
    title="LIBERTY-conditioned word vectors across slices",
    xaxis=dict(dtick=5),
    yaxis_title="Projection along PC1",
    template="plotly_dark",
    height=600,
)

output_file = config.OUT_DIR / "liberty_poles_interactive.html"
fig.write_html(output_file, include_plotlyjs="cdn")
logger.info(f"Wrote {output_file}")
