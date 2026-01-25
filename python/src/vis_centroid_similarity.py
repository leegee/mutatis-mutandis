import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import lib.eebo_db as eebo_db

"""
vis_centroid_similarity.py

Fetches centroids from db,
computes cosine similarity of each slice versus the first slice as anchor (later accepting an arg to determine anchor slice),
then plot against time.
"""

with eebo_db.get_connection() as conn:
    cur = conn.cursor()

    cur.execute("""
        SELECT slice_start, centroid
        FROM concept_slice_stats
        WHERE concept_name = 'LIBERTY'
        ORDER BY slice_start
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

# Prepare centroids array
slice_starts: list[int] = []
centroids: np.ndarray  # <- tell mypy this will be an ndarray

tmp_centroids: list[np.ndarray] = []

for slice_start, centroid_str in rows:
    centroid = np.array(centroid_str)
    slice_starts.append(slice_start)
    tmp_centroids.append(centroid)

centroids = np.stack(tmp_centroids)

# Compute cosine similarity to first slice
anchor = centroids[0].reshape(1, -1)
similarities = cosine_similarity(anchor, centroids).flatten()

# Plot semantic drift
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 8))  # width >= 1280px

ax.plot(slice_starts, similarities, marker='o', color='cyan', linewidth=2)
ax.set_title("Semantic Drift of 'LIBERTY'", fontsize=20, fontweight='bold')
ax.set_xlabel("Slice Start Year", fontsize=14)
ax.set_ylabel("Cosine Similarity to 1625", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, alpha=0.3)

# Highlight min similarity (largest semantic change)
min_idx = np.argmin(similarities)
ax.annotate(f"Max drift: {slice_starts[min_idx]}",
            xy=(slice_starts[min_idx], similarities[min_idx]),
            xytext=(slice_starts[min_idx]+2, similarities[min_idx]-0.05),
            color='yellow', fontsize=12, fontweight='bold',
            arrowprops=dict(facecolor='yellow', arrowstyle='->'))

plt.show()
