#!/usr/bin/env python
"""
compute_eg_poles.py

Compute synchronic conceptual poles for 'LIBERTY' per slice using FAISS indexes.

Poles are derived from PCA over the slice vocabulary space,
not from liberty centroids.
"""

from __future__ import annotations
from typing import List, Dict
import numpy as np
import faiss
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

import lib.eebo_config as config
from lib.compute_synchronic_poles import compute_synchronic_poles
from build_faiss_slice_indexes import faiss_slice_path, vocab_slice_path

# --- Stopwords ---
nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
EEBO_EXTRA = {
    "s", "thou", "thee", "thy", "thine", "hath", "doth", "art", "ye", "v",
    "may", "shall", "upon", "us", "yet", "would", "one", "unto", "said", "de",
    "c", "also", "do", "day", "bee", "be", "doe", "therefore"
}
STOPWORDS.update(EEBO_EXTRA)

TOP_N_WORDS = 20

# --- Prepare slices ---
slice_starts: List[int] = []
pos_words_per_slice: List[List[str]] = []
neg_words_per_slice: List[List[str]] = []

all_words_set = set()

for slice_range in config.SLICES:
    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)
    if not index_path.exists() or not vocab_path.exists():
        continue

    print(f"\n=== Slice {slice_range[0]}–{slice_range[1]} ===")

    index = faiss.read_index(str(index_path))
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    # --- Build vector matrix for entire slice vocabulary ---
    vecs = []
    words = []

    for i, token in enumerate(vocab):
        if token in STOPWORDS:
            continue
        vec = index.reconstruct(i)
        # discard frequency, keep semantic direction
        vec /= np.linalg.norm(vec)
        vecs.append(vec)
        words.append(token)

    if len(vecs) < 50:
        print("Too few vectors in slice, skipping.")
        continue

    vecs_array = np.stack(vecs).astype(np.float32)

    poles = compute_synchronic_poles(
        word_vectors=vecs_array,
        words=words,
        top_n=TOP_N_WORDS,
        stopwords=STOPWORDS
    )

    pos_slice = [w for w, _ in poles["positive_pole"]]
    neg_slice = [w for w, _ in poles["negative_pole"]]

    print(f"Explained variance PC1: {poles['explained_variance']:.3f}")

    print("Positive pole:", pos_slice)
    print("Negative pole:", neg_slice)

    slice_starts.append(slice_range[0])
    pos_words_per_slice.append(pos_slice)
    neg_words_per_slice.append(neg_slice)

    all_words_set.update(pos_slice + neg_slice)

if not slice_starts:
    print("No valid slices found.")
    exit(0)

# --- Build rank arrays for plotting ---
all_words = sorted(all_words_set)
word_to_idx: Dict[str, int] = {w: i for i, w in enumerate(all_words)}

num_slices = len(slice_starts)
pos_array = np.full((num_slices, len(all_words)), np.nan)
neg_array = np.full((num_slices, len(all_words)), np.nan)

for i, (pos_slice, neg_slice) in enumerate(zip(pos_words_per_slice, neg_words_per_slice, strict=True)):
    for rank, word in enumerate(pos_slice):
        pos_array[i, word_to_idx[word]] = rank + 1
    for rank, word in enumerate(neg_slice):
        neg_array[i, word_to_idx[word]] = rank + 1

# --- Plot ---
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 8))

cmap = plt.get_cmap("tab20")
colors = cmap(np.linspace(0, 1, len(all_words)))

for idx, word in enumerate(all_words):
    color = colors[idx]

    ax.plot(slice_starts, pos_array[:, idx], 'o-', color=color)
    ax.text(slice_starts[-1] + 1, pos_array[-1, idx], f"{word} (+)", color=color, fontsize=10, va='center')

    ax.plot(slice_starts, neg_array[:, idx], 'x--', color=color)
    ax.text(slice_starts[-1] + 1, neg_array[-1, idx], f"{word} (-)", color=color, fontsize=10, va='center')

ax.set_xlabel("Slice start year", fontsize=14)
ax.set_ylabel("Rank (1 = strongest pole word)", fontsize=14)
ax.set_title("Synchronic conceptual poles structuring LIBERTY per slice", fontsize=16)
ax.invert_yaxis()

plt.tight_layout()
plt.show()
