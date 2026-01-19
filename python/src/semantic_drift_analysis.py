#!/usr/bin/env python
"""
semantic_drift_analysis.py

Track semantic drift of a canonical term across EEBO temporal slices using
slice-wise fastText models. Produces a large heatmap of nearest-neighbour similarities.
"""

import fasttext
from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.eebo_config as config
from build_spelling_map_from_fastText import get_canonical_spelling_map

# Load canonical spelling map
spelling_map = get_canonical_spelling_map(dry=True)

def canonicalise(term: str) -> str:
    """Map historical variant to canonical form."""
    return spelling_map.get(term, term)

def load_slice_model(slice_name: str) -> Any:
    """Load fastText model for a given slice."""
    model_path = config.MODELS_DIR / f"{slice_name}.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Slice model not found: {model_path}")
    return fasttext.load_model(str(model_path))

def get_slice_neighbors(model, term: str, k: int = 10):
    """Return k nearest neighbors and similarity scores for a term in a slice."""
    try:
        neighbors = model.get_nearest_neighbors(term, k=k)
        return {n: sim for sim, n in neighbors}
    except KeyError:
        # Term not in model vocab
        return {}

def collect_neighbors_across_slices(term: str, slices: list, k: int = 10) -> pd.DataFrame:
    """Build DataFrame of neighbors vs slice with similarity scores."""
    term_canon = canonicalise(term)
    all_data = {}
    for start, end in slices:
        slice_name = f"{start}-{end}"
        model = load_slice_model(slice_name)
        neighbors = get_slice_neighbors(model, term_canon, k=k)
        all_data[slice_name] = neighbors
    # Build DataFrame: rows = neighbors, columns = slices
    df = pd.DataFrame(all_data).fillna(0.0)
    df = df.sort_index()
    return df

def plot_heatmap(df: pd.DataFrame, term: str, figsize=(20, 30), cmap="viridis"):
    """Plot a very large heatmap of semantic drift."""
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".2f", cmap=cmap)
    plt.title(f"Semantic Drift of '{term}' Across Slices")
    plt.xlabel("Temporal Slice")
    plt.ylabel("Nearest Neighbours")
    plt.tight_layout()
    plt.show()

def main():
    term = input("Enter canonical term to analyze: ").strip()
    k = 15  # top neighbors
    df = collect_neighbors_across_slices(term, config.SLICES, k=k)
    print(df)
    plot_heatmap(df, term, figsize=(20, 30))

if __name__ == "__main__":
    main()
