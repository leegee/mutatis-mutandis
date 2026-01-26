#!/usr/bin/env python
"""
vis_centroid_similarity_neighbours.py

Fetch centroids from DB, compute direct slice-to-slice cosine distance,
plot semantic drift, and show nearest neighbours around each centroid
using per-slice FAISS indexes. Also tracks all acceptable forms of a concept
and merges orthographic variants.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import faiss

import lib.eebo_db as eebo_db
from lib import eebo_config as config
from build_faiss_slice_indexes import faiss_slice_path, vocab_slice_path

# Plot settings
FIG_WIDTH = 16
FIG_HEIGHT = 8
MARKER_COLOR = "cyan"
ANNOT_COLOR = "yellow"
NEIGHBOR_COLOR = "orange"
FORM_COLOR = "lime"
TITLE_FONT = 20
LABEL_FONT = 14
TICK_FONT = 12
SHOW_LABELS = False
SHOW_FORMS = False
TOP_K = 10

# Merge threshold for orthographic variants
MERGE_SIM_THRESHOLD = 0.85
LIBERTY_FORMS = config.CONCEPT_SETS["LIBERTY"]["forms"]

# Fetch centroids from DB
with eebo_db.get_connection() as conn:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT slice_start, centroid
        FROM concept_slice_stats
        WHERE concept_name = 'LIBERTY'
        ORDER BY slice_start
        """
    )
    rows: List[Tuple[int, Union[str, List[float]]]] = cur.fetchall()
    cur.close()

slice_starts: List[int] = []
centroids_list: List[np.ndarray] = []

for slice_start, centroid_obj in rows:
    if isinstance(centroid_obj, str):
        centroid = np.array([float(x) for x in centroid_obj.strip("{}").split(",")], dtype=np.float32)
    else:
        centroid = np.array(centroid_obj, dtype=np.float32)
    slice_starts.append(slice_start)
    centroids_list.append(centroid)

centroids: np.ndarray = np.stack(centroids_list)  # shape: (num_slices, dim)

# Normalize centroids
centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

# Initialize merged_centroids
merged_centroids: List[np.ndarray] = []

# Prepare plot
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# Initialize neighbor and form tracking
neighbor_sims_list: List[List[float]] = []
neighbor_labels_list: List[List[str]] = []
form_sims: Dict[str, List[float]] = {form: [] for form in LIBERTY_FORMS}
slice_vecs_list: List[Optional[np.ndarray]] = []
merge_log: List[Dict[str, list]] = []

print("\n=== LIBERTY neighbor variants by slice (merged centroid) ===\n")

for i, slice_start in enumerate(slice_starts):
    slice_range = next((s for s in config.SLICES if s[0] <= slice_start <= s[1]), None)
    if slice_range is None:
        neighbor_sims_list.append([])
        neighbor_labels_list.append([])
        for form in form_sims:
            form_sims[form].append(np.nan)
        merged_centroids.append(centroids[i])
        merge_log.append({"accepted": [], "rejected": []})
        continue

    index_path = faiss_slice_path((slice_range[0], slice_range[1]))
    vocab_path = vocab_slice_path((slice_range[0], slice_range[1]))
    if not index_path.exists() or not vocab_path.exists():
        neighbor_sims_list.append([])
        neighbor_labels_list.append([])
        for form in form_sims:
            form_sims[form].append(np.nan)
        merged_centroids.append(centroids[i])
        merge_log.append({"accepted": [], "rejected": []})
        continue

    index = faiss.read_index(str(index_path))
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    # Normalize canonical vector
    canonical_vec = centroids[i].astype(np.float32)

    # Merge orthographic variants
    merge_vecs = [canonical_vec]
    accepted_forms = []
    rejected_forms = []

    for form in LIBERTY_FORMS:
        if form in vocab and form != "liberty":
            idx = vocab.index(form)
            vec = index.reconstruct(idx)
            vec /= np.linalg.norm(vec)
            sim = float(np.dot(canonical_vec, vec))
            if sim >= MERGE_SIM_THRESHOLD:
                merge_vecs.append(vec)
                accepted_forms.append(form)
            else:
                rejected_forms.append((form, sim))

    merged_centroid = np.mean(merge_vecs, axis=0)
    merged_centroid /= np.linalg.norm(merged_centroid)

    merged_centroids.append(merged_centroid)
    merge_log.append({"accepted": accepted_forms, "rejected": rejected_forms})

    # Print merge summary
    print(f"Slice {slice_start} ({slice_range[0]}-{slice_range[1]}):")
    print(f"  Accepted variants: {accepted_forms}")
    if rejected_forms:
        print(f"  Rejected variants: {[f'{f} ({s:.3f})' for f,s in rejected_forms]}")
    print()

    # Query FAISS neighbors using merged centroid
    distances, indices = index.search(merged_centroid.reshape(1, -1), TOP_K + len(LIBERTY_FORMS))
    sims = distances[0].tolist()
    tokens = [vocab[idx] for idx in indices[0]]

    # Exclude all merged forms from neighbor results
    filtered_tokens_sims = [(t, s) for t, s in zip(tokens, sims, strict=True) if t not in LIBERTY_FORMS]
    if filtered_tokens_sims:
        unzipped_tokens, unzipped_sims = zip(*filtered_tokens_sims, strict=True)
        filtered_tokens_list = list(unzipped_tokens[:TOP_K])
        filtered_sims_list = list(unzipped_sims[:TOP_K])
    else:
        filtered_tokens_list = []
        filtered_sims_list = []

    neighbor_labels_list.append(filtered_tokens_list)
    neighbor_sims_list.append(filtered_sims_list)

    # PRINT neighbors
    print("  Top neighbors (excluding merged forms):")
    for t, s in zip(filtered_tokens_list, filtered_sims_list, strict=True):
        print(f"    {t} ({s:.3f})")
    print()

    # Track per-form similarity against merged centroid
    for form in form_sims:
        if form in vocab:
            idx = vocab.index(form)
            vec = index.reconstruct(idx)
            vec /= np.linalg.norm(vec)
            form_sims[form].append(float(np.dot(merged_centroid, vec)))
        else:
            form_sims[form].append(np.nan)

    if accepted_forms:
        slice_vecs_list.append(
            np.mean(
                [index.reconstruct(vocab.index(f))/np.linalg.norm(index.reconstruct(vocab.index(f))) for f in accepted_forms],
                axis=0
            )
        )
    else:
        slice_vecs_list.append(None)

# --- Slice-to-slice cosine distance ---
slice_to_slice_distances: List[float] = []

for i in range(len(merged_centroids) - 1):
    cos_sim = float(np.dot(merged_centroids[i], merged_centroids[i+1]))
    cos_dist = 1 - cos_sim
    slice_to_slice_distances.append(cos_dist)

slice_midpoints = [(slice_starts[i] + slice_starts[i+1]) / 2 for i in range(len(slice_starts)-1)]

# Slice-to-slice drift table
print("\n=== Slice-to-slice semantic drift (cosine distance) ===\n")
print(f"{'From':>6} -> {'To':>6} | {'CosineDist':>10}")
print("-" * 30)
for i in range(len(slice_to_slice_distances)):
    print(f"{slice_starts[i]:>6} -> {slice_starts[i+1]:>6} | {slice_to_slice_distances[i]:>10.4f}")

# Plot as magenta bars on secondary y-axis
ax2 = ax.twinx()
ax2.bar(slice_midpoints, slice_to_slice_distances, width=3, color="magenta", alpha=0.5, label="Slice-to-slice drift")
ax2.set_ylabel("Cosine distance to previous slice", fontsize=LABEL_FONT, color="magenta")
ax2.tick_params(axis="y", labelsize=TICK_FONT, colors="magenta")

plt.tight_layout()
plt.show()
