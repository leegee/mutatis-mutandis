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

    for slice_key, concepts in data.items():
        for concept, clusters in concepts.items():
            for cluster_id, cluster_data in clusters.items():
                masses = cluster_data.get("mass", {})
                tokens = cluster_data.get("tokens", [])

                rows.append({
                    "Slice": slice_key,
                    "Concept": concept,
                    "Cluster": str(cluster_id),
                    "Tokens": ", ".join(tokens),

                    # 🔹 all mass modes
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


def plot_interactive(df, concept, output_html):
    subset = df[df["Concept"] == concept]

    MASS_MODES = ["count", "freq", "sim", "weighted"]
    pivots = {}
    hover_texts = {}

    #  matrices for every mass definition
    for mode in MASS_MODES:
        pivots[mode] = (
            subset
            .pivot(index="Cluster", columns="Slice", values=f"Mass_{mode}")
            .fillna(0)
        )

        hover_texts[mode] = (
            subset
            .pivot(index="Cluster", columns="Slice", values="Tokens")
            .fillna("")
        )

    initial_mode = "weighted"

    fig = px.imshow(
        pivots[initial_mode].values,
        x=pivots[initial_mode].columns,
        y=pivots[initial_mode].index,
        aspect="auto",
        labels=dict(
            x="Slice",
            y="Cluster ID",
            color=f"Cluster mass ({initial_mode})",
        ),
    )

    fig.update_traces(
        customdata=hover_texts[initial_mode].values,
        hovertemplate=(
            "Cluster ID: %{y}<br>"
            "Slice: %{x}<br>"
            f"Mass ({initial_mode}): %{{z}}<br>"
            "<br><b>Tokens</b><br>%{customdata}"
        ),
    )

    # dropdown to swap mass mode
    fig.update_layout(
        title=f"Cluster mass of '{concept}' ({initial_mode})",
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label=mode,
                        method="update",
                        args=[
                            {
                                "z": [pivots[mode].values],
                                "customdata": [hover_texts[mode].values],
                            },
                            {
                                "coloraxis.colorbar.title": f"Cluster mass ({mode})",
                                "title": f"Cluster mass of '{concept}' ({mode})",
                            },
                        ],
                    )
                    for mode in MASS_MODES
                ],
                direction="down",
                showactive=True,
                x=1.02,
                y=1.0,
            )
        ],
    )

    fig.write_html(output_html)


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
