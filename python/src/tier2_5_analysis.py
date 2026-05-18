#!/usr/bin/env python
"""
tier2_5_analysis.py

Analytical layer over Tier 2.5 concept neighbour output.

This script converts instance-level semantic neighbourhoods into:
    1. slice x neighbour field matrices
    2. temporal drift curves
    3. neighbour turnover metrics

RATIONALE

Tier 2.5 preserves:
    - individual semantic events (instances)
    - embedding geometry (no modification)
    - slice provenance (temporal structure via Zarr partitioning)

This layer treats those instances as observations of:
    a time-indexed semantic field per concept

We explicitly avoid:
    - centroid collapse across slices
    - global normalisation across concepts
    - loss of neighbour identity structure

Instead we compute:
    - within-slice distributions
    - between-slice geometry changes
    - set-based vocabulary turnover
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from lib.eebo_config import OUT_DIR


INPUT_PATH = OUT_DIR / "tier2_5_concept_neighbours_temporal.json"


# Load + flatten
def load_data():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def to_dataframe(data):
    rows = []

    for concept, payload in data["concepts"].items():
        if payload.get("empty"):
            continue

        for inst in payload["instances"]:
            slice_id = inst.get("slice")

            for neigh in inst["neighbours"]:
                rows.append({
                    "concept": concept,
                    "slice": slice_id,
                    "source_token": inst["token"],
                    "neighbour_token": neigh["token"],
                    "similarity": neigh["similarity"]
                })

    df = pd.DataFrame(rows).dropna(subset=["slice"])
    return df


# Slice x neighbour matrix
def build_matrix(df, concept, top_n=30):
    sub = df[df["concept"] == concept]

    agg = (
        sub.groupby(["slice", "neighbour_token"])
           .agg(
               count=("similarity", "size"),
               mean_sim=("similarity", "mean")
           )
           .reset_index()
    )

    agg["strength"] = agg["count"] * agg["mean_sim"]

    top = (
        agg.groupby("neighbour_token")["strength"]
           .sum()
           .sort_values(ascending=False)
           .head(top_n)
           .index
    )

    agg = agg[agg["neighbour_token"].isin(top)]

    matrix = (
        agg.pivot(
            index="slice",
            columns="neighbour_token",
            values="strength"
        )
        .fillna(0)
    )

    # per-slice normalisation
    matrix = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

    return matrix


# Heatmap
def plot_heatmap(matrix, concept):
    plt.figure(figsize=(14, 6))
    plt.imshow(matrix.values, aspect="auto", cmap="magma_r")
    plt.yticks(range(len(matrix.index)), matrix.index)
    plt.xticks(
        range(len(matrix.columns)),
        matrix.columns,
        rotation=45,
        ha="right"
    )
    plt.title(f"{concept} — Slice Semantic Field")
    plt.colorbar(label="normalised neighbour strength")
    plt.tight_layout()
    plt.show()


# Drift curve (cosine change between slices)
def drift_curve(matrix):
    vectors = matrix.values
    slices = matrix.index.tolist()
    drifts = [0.0]

    for i in range(1, len(vectors)):
        sim = cosine_similarity(
            vectors[i - 1].reshape(1, -1),
            vectors[i].reshape(1, -1)
        )[0, 0]

        drifts.append(1 - sim)

    return pd.DataFrame({
        "slice": slices,
        "drift": drifts
    })


def plot_drift(matrix, concept):
    d = drift_curve(matrix)
    plt.figure(figsize=(10, 4))
    plt.plot(d["slice"], d["drift"], marker="o")
    plt.xticks(rotation=45)
    plt.title(f"{concept} — Semantic Drift Between Slices")
    plt.ylabel("1 - cosine similarity")
    plt.xlabel("Slice")
    plt.tight_layout()
    plt.show()
    return d


# Turnover (set-based neighbour change)
def turnover(matrix, top_k=10):
    slices = matrix.index
    results = [0.0]
    prev = None

    for i, row in enumerate(matrix.values):
        top = set(
            matrix.columns[np.argsort(row)[-top_k:]]
        )

        if prev is None:
            prev = top
            continue

        union = prev | top
        inter = prev & top
        score = 1 - (len(inter) / len(union) if union else 0.0)
        results.append(score)
        prev = top

    return pd.DataFrame({
        "slice": slices,
        "turnover": results
    })


def plot_turnover(matrix, concept):
    t = turnover(matrix)
    plt.figure(figsize=(10, 4))
    plt.plot(t["slice"], t["turnover"], marker="o")
    plt.xticks(rotation=45)
    plt.title(f"{concept} — Neighbour Turnover")
    plt.ylabel("1 - Jaccard similarity")
    plt.xlabel("Slice")
    plt.tight_layout()
    plt.show()
    return t


# Full pipeline for one concept
def analyse_concept(df, concept):
    matrix = build_matrix(df, concept)
    plot_heatmap(matrix, concept)
    drift = plot_drift(matrix, concept)
    turn = plot_turnover(matrix, concept)
    return {
        "matrix": matrix,
        "drift": drift,
        "turnover": turn
    }


# Run all concepts
def main():
    data = load_data()
    df = to_dataframe(data)
    concepts = df["concept"].unique()
    results = {}
    for c in concepts:
        print(f"[analysis] processing {c}")
        results[c] = analyse_concept(df, c)
    return results


if __name__ == "__main__":
    main()
