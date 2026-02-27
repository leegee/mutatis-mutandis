# secularisation_analysis.py

"""
Project the movement of CONCEPTS' canonical centroids against semantic axes
using cosine similarity.
"""

from typing import Dict, List
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

from lib.eebo_config import SLICES, CONCEPT_SETS, ALIGNED_VECTORS_DIR

# Minimum frequency threshold to include a form in centroid
MIN_FREQ = 5


def load_slice_vectors(slice_start, slice_end):
    """
    Load aligned FastText vectors for a slice.
    Assumes JSON file: ALIGNED_VECTORS_DIR / f"{slice_start}_{slice_end}.json"
    Format: { word: [float, float, ...] }
    Todo: move path to a func
    """
    path = ALIGNED_VECTORS_DIR / f"{slice_start}-{slice_end}.json"
    with open(path, "r", encoding="utf-8") as f:
        vectors = json.load(f)
    # Convert lists to numpy arrays
    return {w: np.array(v) for w, v in vectors.items()}


def compute_centroid(words, vectors):
    """
    Compute centroid of a set of words for a given slice.
    Ignore words not present or below frequency threshold.
    """
    vecs = []
    for w in words:
        if w in vectors:
            vecs.append(vectors[w])
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return np.nan
    return 1 - cosine(v1, v2)


def build_axes(vectors):
    """
    Returns a dict of axes to project words onto.
    Example axes:
      - DIVINE -> TEMPORAL (secularisation)
      - PREROGATIVE -> LIBERTY (political authority)
    """
    axes = {}

    # Secularisation axis
    divine_centroid = compute_centroid(CONCEPT_SETS['DIVINE']['forms'], vectors)
    temporal_centroid = compute_centroid(CONCEPT_SETS['TEMPORAL']['forms'], vectors)
    if divine_centroid is not None and temporal_centroid is not None:
        axes['DIVINE_TEMPORAL'] = temporal_centroid - divine_centroid

    # Political authority axis
    prerogative_centroid = compute_centroid(CONCEPT_SETS['PREROGATIVE']['forms'], vectors)
    liberty_centroid = compute_centroid(CONCEPT_SETS['LIBERTY']['forms'], vectors)
    if prerogative_centroid is not None and liberty_centroid is not None:
        axes['PREROGATIVE_LIBERTY'] = liberty_centroid - prerogative_centroid

    # Optional: KING vs PARLIAMENT axis
    if 'KING' in CONCEPT_SETS and 'PARLIAMENT' in CONCEPT_SETS:
        king_centroid = compute_centroid(CONCEPT_SETS['KING']['forms'], vectors)
        parliament_centroid = compute_centroid(CONCEPT_SETS['PARLIAMENT']['forms'], vectors)
        if king_centroid is not None and parliament_centroid is not None:
            axes['KING_PARLIAMENT'] = parliament_centroid - king_centroid

    return axes


def compute_trajectories():
    """
    Compute per-slice projections for each canonical head.
    Returns:
        trajectories: dict of {concept_head: [per-slice scores]}
    """
    trajectories: Dict[str, List[float]] = {head: [] for head in CONCEPT_SETS}

    for slice_start, slice_end in SLICES:
        print(f"Processing slice {slice_start}-{slice_end} ...")
        vectors = load_slice_vectors(slice_start, slice_end)
        axes = build_axes(vectors)

        for head, info in CONCEPT_SETS.items():
            centroid = compute_centroid(info['forms'], vectors)
            if centroid is None:
                trajectories[head].append(np.nan)
                continue

            # Project onto secularisation axis
            if 'DIVINE_TEMPORAL' in axes:
                score = cosine_similarity(centroid, axes['DIVINE_TEMPORAL'])
            else:
                score = np.nan

            trajectories[head].append(float(score))

    return trajectories


def plot_trajectories(trajectories, save_path: str = config.OUT_DIR / "secularisation_trajectories.png"):
    """
    Plot and save trajectories of canonical concepts along the DIVINE → TEMPORAL axis.
    - High-resolution PNG (2160x1215 px)
    - Distinguishable colors (tab20), line styles, and highlighted key concepts
    """
    fig_width, fig_height = 12, 6.75  # inches
    dpi = 180
    plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    # Hivis
    cmap = plt.get_cmap("tab20")
    linestyles = ['-', '--', ':', '-.']

    # Key concepts
    highlights = ["LIBERTY", "PREROGATIVE"]

    heads = list(trajectories.keys())
    for i, head in enumerate(heads):
        scores = trajectories[head]
        # Apply tiny vertical jitter to dense slices to reduce overlap
        jittered = scores # [s + (i * 0.005) if not np.isnan(s) else np.nan for s in scores]

        # Line style and color
        color = cmap(i / len(heads))
        linestyle = linestyles[i % len(linestyles)]
        linewidth = 3 if head in highlights else 1.5
        alpha = 1.0 if head in highlights else 0.6
        plt.plot(
            [start for start, _ in SLICES],
            jittered,
            label=head,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker='o',
            alpha=alpha
        )

    plt.xlabel("Slice start year")
    plt.ylabel("Projection onto DIVINE → TEMPORAL axis")
    plt.title("Secularisation Trajectories of Canonical Heads")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))  # legend outside plot

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    print(f"Saved secularisation plot to: {save_path}")
    plt.show()


if __name__ == "__main__":
    trajectories = compute_trajectories()
    plot_trajectories(trajectories)

