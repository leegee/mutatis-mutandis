#!/usr/bin/env python
"""
viz_usage_clusters_combined.py

Visualise usage clusters of concepts over time with cluster mass in a single PNG.

Input is the output of `usage_cluster_merged.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lib.eebo_config as config


TARGET: Optional[str] = None

# Which mass definition to visualise:
# "count", "freq", "sim", "weighted"
MASS_MODE = "weighted"

if TARGET:
    IN_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    IN_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

OUT_FILE = config.OUT_DIR / f"cluster_mass_combined_{MASS_MODE}.png"


def load_data(in_file: Path) -> pd.DataFrame:
    """Load merged cluster JSON into a long-format DataFrame."""
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for concept, slices in data.items():
        for slice_key, clusters in slices.items():
            for cluster_id, cluster_data in clusters.items():
                tokens = cluster_data.get("tokens", [])
                masses = cluster_data.get("masses", {})
                mass = masses.get(MASS_MODE, 0)

                rows.append(
                    {
                        "Slice": slice_key,
                        "Concept": concept,
                        "Cluster": cluster_id,
                        "Tokens": ", ".join(tokens),
                        "Mass": mass,
                    }
                )

    return pd.DataFrame(rows)


def sort_cluster_keys(keys):
    """Sort cluster IDs numerically if possible, keeping '-1' (outliers) first."""
    numeric = []
    other = []

    for k in keys:
        try:
            v = int(k)
            if v != -1:
                numeric.append(v)
        except ValueError:
            other.append(k)

    numeric.sort()
    other.sort()

    result = []
    if "-1" in keys:
        result.append("-1")
    result += [str(v) for v in numeric]
    result += [str(v) for v in other]
    return result


def plot_cluster_mass_combined(
    df: pd.DataFrame,
    out_file: Path,
    annot_fontsize: int = 18,
):
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
        axes = [axes]

    for ax, concept in zip(axes, concepts, strict=True):
        subset = df[df["Concept"] == concept]

        pivot = (
            subset
            .pivot(index="Cluster", columns="Slice", values="Mass")
            .fillna(0)
        )

        pivot = pivot.reindex(sort_cluster_keys(pivot.index), axis=0)
        pivot.columns = pivot.columns.astype(str)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f" if MASS_MODE in {"sim", "weighted"} else ".0f",
            cmap="YlGnBu",
            cbar=ax == axes[-1],
            ax=ax,
            annot_kws={"size": annot_fontsize, "weight": "bold"},
            cbar_kws={"label": f"Cluster mass ({MASS_MODE})"} if ax == axes[-1] else None,
            linewidths=0.5,
            linecolor="gray",
        )

        ax.set_ylabel("Cluster ID", fontsize=10, weight="bold")
        ax.set_xlabel("Slice", fontsize=10, weight="bold")
        ax.set_title(f"Concept '{concept}'", fontsize=12, weight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    plt.suptitle(
        f"Cluster mass over time ({MASS_MODE})",
        fontsize=16,
        weight="bold",
    )
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved combined figure to {out_file}")


def main():
    df = load_data(IN_FILE)
    print(f"Loaded {len(df)} cluster entries from {IN_FILE}")
    plot_cluster_mass_combined(df, OUT_FILE)

if __name__ == "__main__":
    main()
