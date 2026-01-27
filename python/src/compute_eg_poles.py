#!/usr/bin/env python
"""
compute_liberty_poles.py

Compute conceptual poles for 'LIBERTY' per slice using FAISS indexes
and merged centroids from vis_centroid_similarity_neighbours.py.

Outputs top words for positive and negative PC1 poles per slice.

WIP. Possibly a dead end.
"""

from __future__ import annotations
from typing import List
import numpy as np
import faiss

import nltk
from nltk.corpus import stopwords

import lib.eebo_db as eebo_db
import lib.eebo_config as config
from lib.compute_conceptual_poles import compute_conceptual_poles
from build_faiss_slice_indexes import faiss_slice_path, vocab_slice_path

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
EEBO_EXTRA = {
    "s", "thou", "thee", "thy", "thine", "hath", "doth", "art", "ye", "v",
    "may", "shall", "upon", "us", "yet", "would", "one", "unto", "said", "de"
}
STOPWORDS.update(EEBO_EXTRA)

TOP_N_WORDS = 15
MERGE_SIM_THRESHOLD = 0.85
LIBERTY_FORMS = config.CONCEPT_SETS["LIBERTY"]["forms"]

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
    rows = cur.fetchall()
    cur.close()

slice_starts: List[int] = []
merged_centroids: List[np.ndarray] = []

for slice_start, centroid_obj in rows:
    if isinstance(centroid_obj, str):
        centroid = np.array([float(x) for x in centroid_obj.strip("{}").split(",")], dtype=np.float32)
    else:
        centroid = np.array(centroid_obj, dtype=np.float32)
    centroid /= np.linalg.norm(centroid)  # normalize
    slice_starts.append(slice_start)
    merged_centroids.append(centroid)

slice_vecs_list: List[np.ndarray] = []
slice_contexts: List[str] = []

for i, slice_start in enumerate(slice_starts):
    # Find slice range
    slice_range = next((s for s in config.SLICES if s[0] <= slice_start <= s[1]), None)

    if slice_range is None:
        # just skip
        continue

    index_path = faiss_slice_path(slice_range)
    vocab_path = vocab_slice_path(slice_range)
    if not index_path.exists() or not vocab_path.exists():
        # skip
        continue

    index = faiss.read_index(str(index_path))
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]

    # Merge orthographic variants
    merge_vecs = [merged_centroids[i]]  # start with canonical vector
    for form in LIBERTY_FORMS:
        if form in vocab and form != "liberty":
            idx = vocab.index(form)
            vec = index.reconstruct(idx)
            vec /= np.linalg.norm(vec)
            sim = float(np.dot(merged_centroids[i], vec))
            if sim >= MERGE_SIM_THRESHOLD:
                merge_vecs.append(vec)

    slice_vec = np.mean(merge_vecs, axis=0)
    slice_vec /= np.linalg.norm(slice_vec)
    slice_vecs_list.append(slice_vec)

    # Build slice context string from vocab
    slice_contexts.append(" ".join(vocab))

# Filter out None vectors
valid_vectors = [v for v in slice_vecs_list if v is not None]
valid_contexts = [c for v, c in zip(slice_vecs_list, slice_contexts, strict=True) if v is not None]

if not valid_vectors:
    print("No valid slices found. Exiting.")
    exit(0)

vectors = np.stack(valid_vectors).astype(np.float32)

# --- Compute conceptual poles ---
poles = compute_conceptual_poles(
    slice_vectors=vectors,
    slice_contexts=valid_contexts,
    top_n=TOP_N_WORDS,
    stopwords=STOPWORDS
)

# --- Output results ---
print("\n=== Conceptual poles for 'LIBERTY' across slices ===\n")
print("Positive pole top words:")
for word, count in poles["positive_pole"]:
    print(f"  {word}: {count}")

print("\nNegative pole top words:")
for word, count in poles["negative_pole"]:
    print(f"  {word}: {count}")
