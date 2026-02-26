#!/usr/bin/env python
"""
vis_centroid_similarity_aligned.py

Compute semantic drift of a canonical concept across slices
using pre-aligned embeddings.

Drift is measured as cosine similarity of each slice centroid
against the first slice (anchor).
"""

from __future__ import annotations
from typing import Iterable, List
import numpy as np
import matplotlib.pyplot as plt

from lib.eebo_config import SLICES, CONCEPT_SETS
from align import load_aligned_vectors


CONCEPT: str = "LIBERTY"  # Change to any key in CONCEPT_SETS


def compute_centroid(
    vectors: dict[str, np.ndarray],
    forms: Iterable[str]
) -> np.ndarray | None:
    """
    Compute normalized centroid for a concept from its forms.
    Returns None if no forms are found in this slice.
    """
    collected: List[np.ndarray] = []

    for form in forms:
        key = form.lower()
        if key in vectors:
            vec = vectors[key].astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                collected.append(vec / norm)

    if not collected:
        return None

    centroid = np.mean(np.stack(collected), axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm == 0:
        return None

    return centroid / centroid_norm


forms: Iterable[str] = CONCEPT_SETS[CONCEPT]["forms"]

slice_starts: List[int] = []
centroids_list: List[np.ndarray] = []

for start, end in SLICES:
    slice_id = f"{start}-{end}"

    try:
        aligned_vectors = load_aligned_vectors(slice_id)
    except FileNotFoundError:
        continue

    centroid = compute_centroid(aligned_vectors, forms)
    if centroid is not None:
        centroids_list.append(centroid)
        slice_starts.append(start)

if not centroids_list:
    raise RuntimeError(f"No centroid vectors found for concept '{CONCEPT}'")

# Stack into (num_slices, dim)
centroids: np.ndarray = np.stack(centroids_list)

# Anchor = first slice
anchor: np.ndarray = centroids[0].reshape(1, -1)

# Vectorized cosine similarity
cos_sims: np.ndarray = (centroids @ anchor.T).flatten()

#
# Plot
#

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(slice_starts, cos_sims, marker="o", linewidth=2)
ax.set_title(f"Semantic Drift of '{CONCEPT}'", fontsize=20, fontweight="bold")
ax.set_xlabel("Slice Start Year", fontsize=14)
ax.set_ylabel(f"Cosine Similarity to {slice_starts[0]}", fontsize=14)
ax.grid(True, alpha=0.3)

# Highlight maximum drift (minimum cosine similarity)
min_idx = int(np.argmin(cos_sims))
ax.annotate(
    f"Max drift: {slice_starts[min_idx]}",
    xy=(slice_starts[min_idx], float(cos_sims[min_idx])),
    xytext=(slice_starts[min_idx] + 2, float(cos_sims[min_idx]) - 0.05),
    fontsize=12,
    arrowprops=dict(arrowstyle="->"),
)

plt.tight_layout()
plt.show()
