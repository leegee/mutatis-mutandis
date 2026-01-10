#!/usr/bin/env python
"""
PDF visualisations for semantic neighbourhood analysis.

Includes:
- absorption strength distribution
- semantic stability
- liberty vs freedom
- rank–cosine curves
- Ryan-Heuser-style semantic drift plot

All outputs are written to config.OUT_DIR
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import eebo_db
import eebo_config as config

QUERY = "liberty"


def load_data(query):
    conn = eebo_db.dbh

    df = pd.read_sql_query("""
        SELECT
            n.slice_start,
            n.slice_end,
            n.query,
            n.neighbour,
            n.rank,
            n.cosine,
            COALESCE(m.canonical, n.neighbour) AS concept,
            COALESCE(m.concept_type, 'orthographic') AS concept_type
        FROM neighbourhoods n
        LEFT JOIN spelling_map m
            ON n.neighbour = m.variant
        WHERE n.query = ?
          AND COALESCE(m.concept_type, 'orthographic') != 'exclude'
    """, conn, params=(query,))

    conn.close()

    df["slice"] = df["slice_start"].astype(str)
    return df


def save(fig, name):
    out = Path(config.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def plot_absorption_distribution(df):
    concept_means = df.groupby("concept")["cosine"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(concept_means, bins=20)
    ax.set_xlabel("Mean cosine similarity")
    ax.set_ylabel("Number of concepts")
    ax.set_title("Distribution of semantic similarity among collapsed concepts")

    fig.tight_layout()
    save(fig, "absorption_distribution")


def plot_liberty_stability(df):
    liberty_df = df[df["concept"] == "liberty"]
    stability = liberty_df.groupby("slice")["cosine"].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(stability.index, stability.values, marker="o")
    ax.set_xlabel("Slice")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Semantic stability of 'liberty' over time")
    ax.grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "liberty_stability")


def plot_liberty_vs_freedom(df):
    compare = (
        df[df["concept"].isin(["liberty", "freedom"])]
        .groupby(["slice", "concept"])["cosine"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(compare.index, compare["liberty"], marker="o", label="liberty")
    ax.plot(compare.index, compare["freedom"], marker="o", label="freedom")

    ax.set_xlabel("Slice")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Semantic proximity of 'liberty' vs 'freedom'")
    ax.legend()
    ax.grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "liberty_vs_freedom")


def plot_rank_cosine_curves(df):
    slices = sorted(df["slice"].unique())

    fig, axes = plt.subplots(
        nrows=len(slices),
        ncols=1,
        figsize=(6, 2.2 * len(slices)),
        sharex=True
    )

    if len(slices) == 1:
        axes = [axes]

    for ax, sl in zip(axes, slices):
        subset = df[df["slice"] == sl].sort_values("rank")
        ax.plot(subset["rank"], subset["cosine"], marker=".")
        ax.set_ylabel("Cosine")
        ax.set_title(f"Slice {sl}")

    axes[-1].set_xlabel("Rank")

    fig.tight_layout()
    save(fig, "rank_cosine_curves")


def plot_semantic_drift(df, top_n=10):
    """
    Shows how the neighbourhood of 'liberty' changes across slices.
    Each line tracks a concept's cosine similarity over time.
    """

    top_concepts = (
        df.groupby("concept")["cosine"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    drift = (
        df[df["concept"].isin(top_concepts)]
        .groupby(["slice", "concept"])["cosine"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for concept in drift.columns:
        ax.plot(
            drift.index,
            drift[concept],
            marker="o",
            linewidth=1,
            alpha=0.8,
            label=concept
        )

    ax.set_xlabel("Slice")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_title("Semantic drift of concepts surrounding 'liberty'")
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small"
    )
    ax.grid(True)

    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, "semantic_drift_liberty")


def main():
    df = load_data(QUERY)

    if df.empty:
        print("[INFO] No data found.")
        return

    plot_absorption_distribution(df)
    plot_liberty_stability(df)
    plot_liberty_vs_freedom(df)
    plot_rank_cosine_curves(df)
    plot_semantic_drift(df)


if __name__ == "__main__":
    main()
