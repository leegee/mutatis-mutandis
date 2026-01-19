#!/usr/bin/env python
"""
semantic_drift_analysis.py

Track semantic drift of canonical terms across EEBO temporal slices using
slice-specific fastText models guided by a global canonical reference.
Produces a large heatmap of nearest-neighbour similarities.
"""

import fasttext
from typing import Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.eebo_config as config
from build_spelling_map_from_fastText import get_canonical_spelling_map

# Load canonical spelling map
spelling_map = get_canonical_spelling_map(dry=True)

def canonicalise(term: str) -> Optional[str]:
    """Map historical variant to canonical form using canonical keyword targets."""
    canon = spelling_map.get(term, term)
    if canon in config.KEYWORDS_TO_NORMALISE:
        return canon
    return None

# Load global model for canonical reference
GLOBAL_MODEL_PATH = config.FASTTEXT_GLOBAL_MODEL_PATH
if not GLOBAL_MODEL_PATH.exists():
    raise FileNotFoundError(f"Global fastText model not found: {GLOBAL_MODEL_PATH}")
global_model = fasttext.load_model(str(GLOBAL_MODEL_PATH))

# Helper to load a slice model
def load_slice_model(slice_name: str):
    path = config.FASTTEXT_SLICE_MODEL_DIR / f"{slice_name}.bin"
    if not path.exists():
        raise FileNotFoundError(f"Slice model not found: {path}")
    return fasttext.load_model(str(path))


def get_slice_neighbors(model: Any, vec: list[float], k: int = 10) -> dict[str, float]:
    """
    Return k nearest neighbors in a slice model given a vector.
    """
    try:
        neighbors = model.get_nearest_neighbors(vec, k=k)
        return {n: sim for sim, n in neighbors}
    except Exception:
        return {}

def collect_neighbors_across_slices(k: int = 10) -> pd.DataFrame:
    """
    Build a DataFrame of canonical keywords vs slices, using the global model
    for canonical reference vectors and slice models for temporal neighbors.
    """
    all_slice_data: dict[str, dict[str, float]] = {}
    vocab: set[str] = set()

    # Precompute canonical vectors from the global model
    canonical_vectors: dict[str, list[float]] = {}
    for term in config.KEYWORDS_TO_NORMALISE.keys():
        term_canon = canonicalise(term)
        if term_canon and term_canon in global_model.get_words():
            vec = global_model.get_word_vector(term_canon)
            canonical_vectors[term_canon] = vec.tolist()  # cast ndarray to list

    # For each slice, compute neighbors
    for start, end in config.SLICES:
        slice_name = f"{start}-{end}"
        slice_model = load_slice_model(slice_name)
        slice_neighbors: dict[str, float] = {}
        for _term, vec in canonical_vectors.items():
            neighbors = get_slice_neighbors(slice_model, vec, k=k)
            slice_neighbors.update(neighbors)
        all_slice_data[slice_name] = slice_neighbors
        vocab |= slice_neighbors.keys()

    # Build DataFrame: rows = neighbor terms, columns = slices
    df = pd.DataFrame(
        index=pd.Index(sorted(vocab), dtype=str),
        columns=pd.Index([f"{s[0]}-{s[1]}" for s in config.SLICES], dtype=str)
    )

    for slice_name, neighbors in all_slice_data.items():
        for word, sim in neighbors.items():
            df.loc[word, slice_name] = sim

    # Fill NaNs with 0.0 for heatmap compatibility and ensure float dtype
    df = df.fillna(0.0).astype(float)

    return df

def plot_heatmap(df: pd.DataFrame, figsize=(20, 30), cmap="viridis"):
    """Plot a very large heatmap of semantic drift."""
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap)
    plt.title("Semantic Drift of Canonical Terms Across Slices")
    plt.xlabel("Temporal Slice")
    plt.ylabel("Nearest Neighbours")
    plt.tight_layout()
    plt.show()

def main():
    k = 15  # top neighbors
    df = collect_neighbors_across_slices(k=k)
    print(df)
    plot_heatmap(df, figsize=(20, 30))

if __name__ == "__main__":
    main()
