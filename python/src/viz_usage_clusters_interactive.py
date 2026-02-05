#!/usr/bin/env python
"""
viz_usage_clusters_interactive.py - Interactive cluster mass heatmap with hover-over lexemes.

See `viz_usage_clusters.py`

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import pandas as pd
import plotly.express as px
import lib.eebo_config as config

# Set to None to visualise all concepts
TARGET: Optional[str] = None

if TARGET:
    IN_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    IN_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

OUT_FILE = config.OUT_DIR / "cluster_mass_interactive.html"


def concept_outfile(concept: str) -> Path:
    slug = concept.lower().replace(" ", "_")
    return config.OUT_DIR / f"cluster_mass_interactive_{slug}.html"


def format_tokens(tokens: list[str], per_line: int = 5) -> str:
    lines = [
        ", ".join(tokens[i:i + per_line])
        for i in range(0, len(tokens), per_line)
    ]
    return "<br>".join(lines)


def load_data(in_file: Path) -> pd.DataFrame:
    """Load cluster JSON into a long-format DataFrame using cluster_mass and lexemes."""
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
                        "Tokens": format_tokens(tokens),
                        "Mass": mass,
                    }
                )
    df = pd.DataFrame(rows)
    return df


def sort_cluster_ids(ids: list[str]) -> list[str]:
    def sort_key(cid: str):
        if cid == "-1":
            return (-1, -1)   # outliers first
        try:
            return (0, int(cid))
        except ValueError:
            return (1, cid)  # fallback lexical
    return sorted(ids, key=sort_key)


def plot_interactive(df: pd.DataFrame):
    """Create an interactive heatmap with hover showing tokens."""
    concepts = df["Concept"].unique() if TARGET is None else [TARGET]

    for concept in concepts:
        subset = df[df["Concept"] == concept]

        pivot = subset.pivot(index="Cluster", columns="Slice", values="Mass").fillna(0)
        hover_text = subset.pivot(index="Cluster", columns="Slice", values="Tokens").fillna("")

        # Ensure string cluster IDs
        pivot.index = pivot.index.astype(str)
        hover_text.index = hover_text.index.astype(str)

        # Order clusters safely
        ordered_clusters = sort_cluster_ids(list(pivot.index))
        pivot = pivot.reindex(ordered_clusters)
        hover_text = hover_text.reindex(ordered_clusters)

        fig = px.imshow(
            pivot.values,
            labels=dict(x="Slice", y="Cluster", color="Mass"),
            x=pivot.columns,
            y=pivot.index,
            text_auto=True,
            aspect="auto",
        )

        fig.update_traces(
            hovertemplate=(
                "Cluster ID: %{y}<br>"
                "Slice: %{x}<br>"
                "Mass: %{z}<br>"
                "Tokens: %{customdata}"
            ),
            customdata=hover_text.values,
        )

        fig.update_layout(
            title=f"Cluster Mass of '{concept}'",
            xaxis_title="Slice",
            yaxis_title="Cluster ID",
        )

        out_filepath = concept_outfile(concept)
        fig.write_html(out_filepath)
        print(f"Saved interactive figure for {concept} to {out_filepath}")


def main():
    df = load_data(IN_FILE)
    print(f"Loaded {len(df)} cluster entries from {IN_FILE}")
    plot_interactive(df)


if __name__ == "__main__":
    main()
