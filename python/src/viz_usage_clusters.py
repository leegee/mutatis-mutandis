#!/usr/bin/env python
"""
viz_usage_clusters_combined.py

Visualise usage clusters of concepts over time with cluster mass in a single PNG.

Input is the output of `usage_clusterer_tracker.py`
See also `usage_clusterer.py`.

TODO: combine these files into one module.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.eebo_config as config

# Set to None to visualise all concepts
TARGET: Optional[str] = None

if TARGET:
    IN_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    IN_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

OUT_FILE = config.OUT_DIR / "cluster_mass_combined.png"


def load_data(in_file: Path) -> pd.DataFrame:
    """Load cluster JSON into a long-format DataFrame using precomputed cluster_mass."""
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for concept, slices in data.items():
        for slice_key, slice_data in slices.items():
            clusters = slice_data.get("clusters", {})
            cluster_mass = slice_data.get("cluster_mass", {})
            for cluster_id, tokens in clusters.items():
                mass = cluster_mass.get(cluster_id, len(tokens))
                rows.append(
                    {
                        "Slice": slice_key,
                        "Concept": concept,
                        "Cluster": cluster_id,
                        "Tokens": ", ".join(tokens),
                        "Mass": mass,
                    }
                )
    df = pd.DataFrame(rows)
    return df


def sort_cluster_keys(keys):
    """Sort cluster IDs numerically if possible, keeping '-1' (outliers) at the top."""
    numeric_keys = []
    non_numeric_keys = []
    for k in keys:
        try:
            val = int(k)
            if val == -1:
                continue
            numeric_keys.append(val)
        except ValueError:
            non_numeric_keys.append(k)
    numeric_keys.sort()
    non_numeric_keys.sort()
    result = ["-1"] if "-1" in keys else []
    result += [str(k) for k in numeric_keys]
    result += [str(k) for k in non_numeric_keys]
    return result


def plot_cluster_mass_combined(df: pd.DataFrame, out_file: Path, annot_fontsize: int = 18):
    """Plot cluster mass heatmaps for all concepts in one figure."""
    concepts = df["Concept"].unique() if TARGET is None else [TARGET]

    n_concepts = len(concepts)
    fig, axes = plt.subplots(
        nrows=n_concepts,
        ncols=1,
        figsize=(14, max(4, 4 * n_concepts)),
        constrained_layout=True,
    )

    if n_concepts == 1:
        axes = [axes]  # ensure axes is iterable

    for ax, concept in zip(axes, concepts, strict=True):
        subset = df[df["Concept"] == concept]

        pivot = subset.pivot(index="Cluster", columns="Slice", values="Mass").fillna(0)

        pivot = pivot.reindex(sort_cluster_keys(pivot.index), axis=0)
        pivot.columns = pivot.columns.astype(str)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            cbar=ax == axes[-1],  # show cbar only on last subplot
            ax=ax,
            annot_kws={"size": annot_fontsize, "weight": "bold"},
            cbar_kws={"label": "Cluster Mass"} if ax == axes[-1] else None,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_ylabel("Cluster ID", fontsize=10, weight="bold")
        ax.set_xlabel("Slice", fontsize=10, weight="bold")
        ax.set_title(f"Concept '{concept}'", fontsize=12, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    plt.suptitle("Cluster Mass over Time", fontsize=16, weight="bold")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved combined figure to {out_file}")


def main():
    df = load_data(IN_FILE)
    print(f"Loaded {len(df)} cluster entries from {IN_FILE}")
    plot_cluster_mass_combined(df, OUT_FILE)


if __name__ == "__main__":
    main()
