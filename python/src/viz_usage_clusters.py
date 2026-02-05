#!/usr/bin/env python
"""
viz_usage_clusters.py

Visualise usage clusters of concepts over time with cluster mass.

Requires: pandas, matplotlib, seaborn, numpy
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

def load_data(in_file: Path) -> pd.DataFrame:
    """Load cluster JSON into a long-format DataFrame."""
    with open(in_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for slice_key, clusters in data.items():
        for concept, slice_clusters in clusters.items():
            for cluster_id, tokens in slice_clusters.items():
                rows.append(
                    {
                        "Slice": slice_key,
                        "Concept": concept,
                        "Cluster": cluster_id,
                        "Tokens": ", ".join(tokens),
                        "Mass": len(tokens),
                    }
                )
    df = pd.DataFrame(rows)
    return df


def plot_cluster_mass(df: pd.DataFrame, out_dir: Path):
    """Plot cluster mass heatmap over slices and clusters."""
    concepts = df["Concept"].unique() if TARGET is None else [TARGET]

    for concept in concepts:
        subset = df[df["Concept"] == concept]

        # Pivot table: clusters as rows, slices as columns, mass as values
        pivot = subset.pivot(index="Cluster", columns="Slice", values="Mass").fillna(0)

        plt.figure(figsize=(12, max(4, len(pivot) * 0.5)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            cbar_kws={"label": "Cluster Mass"},
        )
        plt.title(f"Cluster Mass over Time: {concept}")
        plt.ylabel("Cluster ID")
        plt.xlabel("Slice")
        plt.tight_layout()

        # Hover-text replacement: use tokens in annotations if desired
        # (here we just show mass to keep it simple)
        out_path = out_dir / f"cluster_mass_{concept.lower()}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


def main():
    out_dir = config.OUT_DIR
    df = load_data(IN_FILE)
    print(f"Loaded {len(df)} cluster entries from {IN_FILE}")

    plot_cluster_mass(df, out_dir)


if __name__ == "__main__":
    main()
