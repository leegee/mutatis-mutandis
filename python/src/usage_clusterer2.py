#!/usr/bin/env python
"""
viz_usage_clusters_interactive.py

Interactive cluster mass heatmap with hover-over lexemes.

Input is the output of `usage_cluster2.py`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import json
import pandas as pd
import plotly.express as px
from lib.eebo_logging import logger
import lib.eebo_config as config

TARGET: Optional[str] = None  # Set to a concept or None for all

# Mass modes available in your JSON
MASS_MODES = ["count", "freq", "sim", "weighted"]

if TARGET:
    IN_FILE = config.OUT_DIR / f"usage_clusters_{TARGET.lower()}.json"
else:
    IN_FILE = config.OUT_DIR / "usage_clusters_all_concepts.json"


def concept_outfile(concept: str) -> Path:
    slug = concept.lower().replace(" ", "_")
    return config.OUT_DIR / f"cluster_mass_interactive_{slug}.html"


def format_tokens(tokens: list[str], per_line: int = 5) -> str:
    """Pretty-print tokens for hover text."""
    lines = [", ".join(tokens[i:i + per_line]) for i in range(0, len(tokens), per_line)]
    return "<br>".join(lines)


def load_data(json_path: Path) -> pd.DataFrame:
    """Load JSON output into a long-format DataFrame with separate mass columns."""
    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for concept, slices in data.items():
        for slice_key, clusters in slices.items():
            for cluster_id, cluster_data in clusters.items():
                tokens = cluster_data.get("tokens", [])
                masses = cluster_data.get("masses", {})
                rows.append({
                    "Concept": concept,
                    "Slice": slice_key,
                    "Cluster": str(cluster_id),
                    "Tokens": format_tokens(tokens),
                    "Mass_count": masses.get("count", 0),
                    "Mass_freq": masses.get("freq", 0),
                    "Mass_sim": masses.get("sim", 0),
                    "Mass_weighted": masses.get("weighted", 0),
                })
    df = pd.DataFrame(rows)
    return df


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


def plot_interactive(df: pd.DataFrame, concept: str, output_html: Path):
    subset = df[df["Concept"] == concept]

    # Pivot tables for each mass mode
    pivots = {}
    hover_texts = {}
    for mode in MASS_MODES:
        pivots[mode] = subset.pivot_table(
            index="Cluster",
            columns="Slice",
            values=f"Mass_{mode}",
            aggfunc="first"
        ).fillna(0).astype(float)

        hover_texts[mode] = subset.pivot_table(
            index="Cluster",
            columns="Slice",
            values="Tokens",
            aggfunc="first"
        ).fillna("")

        # Order clusters consistently
        pivots[mode] = pivots[mode].reindex(sort_cluster_ids(list(pivots[mode].index)))
        hover_texts[mode] = hover_texts[mode].reindex(sort_cluster_ids(list(hover_texts[mode].index)))

        # Order slices chronologically
        pivots[mode] = pivots[mode].reindex(sorted(pivots[mode].columns), axis=1)
        hover_texts[mode] = hover_texts[mode].reindex(sorted(hover_texts[mode].columns), axis=1)

    # Initial mode to display
    initial_mode = "weighted"

    fig = px.imshow(
        pivots[initial_mode].values,
        x=pivots[initial_mode].columns,
        y=pivots[initial_mode].index,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        labels=dict(x="Slice", y="Cluster ID", color=f"Cluster mass ({initial_mode})"),
    )

    fig.update_traces(
        customdata=hover_texts[initial_mode].values,
        hovertemplate=(
            "Cluster ID: %{y}<br>"
            "Slice: %{x}<br>"
            f"Mass ({initial_mode}): %{{z}}<br>"
            "<br><b>Tokens</b><br>%{customdata}"
        )
    )

    # Dropdown to switch mass mode
    fig.update_layout(
        title=f"Cluster mass of '{concept}' ({initial_mode})",
        updatemenus=[dict(
            buttons=[
                dict(
                    label=mode,
                    method="update",
                    args=[
                        {"z": [pivots[mode].values], "customdata": [hover_texts[mode].values]},
                        {"coloraxis.colorbar.title": f"Cluster mass ({mode})",
                         "title": f"Cluster mass of '{concept}' ({mode})"}
                    ],
                ) for mode in MASS_MODES
            ],
            direction="down",
            showactive=True,
            x=1.02,
            y=1.0,
        )]
    )

    fig.write_html(output_html)
    logger.info(f"Saved interactive plot for {concept} to {output_html}")


def main():
    df = load_data(IN_FILE)
    logger.info(f"Loaded {len(df)} cluster entries from {IN_FILE}")

    for concept in df["Concept"].unique():
        output_html = concept_outfile(concept)
        logger.info(f"Generating interactive plot for concept '{concept}' -> {output_html}")
        plot_interactive(df, concept, output_html)

    logger.info("All interactive plots generated.")


if __name__ == "__main__":
    main()
