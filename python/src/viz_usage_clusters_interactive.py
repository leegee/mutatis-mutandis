#!/usr/bin/env python
"""
viz_usage_clusters_interactive.py

Interactive cluster mass heatmap with hover-over lexemes.

Input is the output of `usage_cluster_merged.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json
import pandas as pd
import plotly.express as px

import lib.eebo_config as config
from lib.eebo_logging import logger


TARGET: Optional[str] = None

# Which mass definition to visualise:
# "count", "freq", "sim", "weighted"
MASS_MODE = "weighted"

if TARGET:
    IN_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    IN_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"

OUT_FILE = config.OUT_DIR / f"cluster_mass_interactive_{MASS_MODE}.html"


def concept_outfile(concept: str) -> Path:
    slug = concept.lower().replace(" ", "_")
    return config.OUT_DIR / f"cluster_mass_interactive_{slug}_{MASS_MODE}.html"


def format_tokens(tokens: list[str], per_line: int = 5) -> str:
    """Pretty-print tokens for hover text."""
    lines = [
        ", ".join(tokens[i:i + per_line])
        for i in range(0, len(tokens), per_line)
    ]
    return "<br>".join(lines)


def load_data(json_path):
    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for concept, slices in data.items():            # top-level keys are concepts
        for slice_key, clusters in slices.items(): # slices / time periods
            for cluster_id, cluster_data in clusters.items():
                masses = cluster_data.get("masses", {})
                tokens = cluster_data.get("tokens", [])

                rows.append({
                    "Slice": slice_key,
                    "Concept": concept,
                    "Cluster": str(cluster_id),
                    "Tokens": format_tokens(tokens),
                    "Mass_count": masses.get("count", 0),
                    "Mass_freq": masses.get("freq", 0),
                    "Mass_sim": masses.get("sim", 0),
                    "Mass_weighted": masses.get("weighted", 0),
                })

    return pd.DataFrame(rows)


def sort_cluster_ids(ids: list[str]) -> list[str]:
    """Sort cluster IDs numerically, keeping '-1' (outliers) first."""
    def sort_key(cid: str):
        if cid == "-1":
            return (-1, -1)
        try:
            return (0, int(cid))
        except ValueError:
            return (1, cid)

    return sorted(ids, key=sort_key)


def plot_interactive_per_mass(df, concept):
    """
    Create an interactive heatmap per mass type for a given concept.
    Each mass type gets its own HTML file and color scale.
    """
    subset = df[df["Concept"] == concept]
    MASS_MODES = ["count", "freq", "sim", "weighted"]

    for mode in MASS_MODES:
        pivot = subset.pivot(index="Cluster", columns="Slice", values=f"Mass_{mode}").fillna(0)
        hover = subset.pivot(index="Cluster", columns="Slice", values="Tokens").fillna("")

        # sort clusters and slices
        ordered_clusters = sort_cluster_ids(list(pivot.index))
        pivot = pivot.reindex(ordered_clusters)
        hover = hover.reindex(ordered_clusters)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        hover = hover.reindex(sorted(hover.columns), axis=1)

        fig = px.imshow(
            pivot.values,
            x=pivot.columns,
            y=pivot.index,
            aspect="auto",
            labels=dict(
                x="Slice",
                y="Cluster ID",
                color=f"Cluster mass ({mode})"
            ),
            zmin=pivot.values.min(),
            zmax=pivot.values.max()
        )

        fig.update_traces(
            customdata=hover.values,
            hovertemplate=(
                "Cluster ID: %{y}<br>"
                "Slice: %{x}<br>"
                f"Mass ({mode}): %{{z}}<br>"
                "<br><b>Tokens</b><br>%{customdata}"
            )
        )

        # write HTML per mass mode
        out_file = config.OUT_DIR / f"cluster_mass_interactive_{concept.lower().replace(' ', '_')}_{mode}.html"
        fig.write_html(out_file)
        logger.info(f"Saved interactive plot for concept '{concept}' mass '{mode}' to {out_file}")


def main():
    df = load_data(IN_FILE)
    logger.info(f"Loaded {len(df)} cluster entries from {IN_FILE}")

    for concept in df["Concept"].unique():
        logger.info(f"Generating interactive plots for concept '{concept}'")
        plot_interactive_per_mass(df, concept)

    logger.info("All interactive plots generated.")


if __name__ == "__main__":
    main()
