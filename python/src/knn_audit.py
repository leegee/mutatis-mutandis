#!/usr/bin/env python
"""
vis_slice_concept_heatmaps_json.py

For each concept and each temporal slice:
- Pull canonical embedding
- Pull nearest neighbors / variant embeddings
- Compute cosine similarity
- Save JSON dump of neighbors + similarities
- Plot a heatmap with text labels, normalized colors, and readable layout
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Tuple, List, Dict

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger
from scipy.spatial.distance import cosine

HEATMAP_DIR = config.OUT_DIR / "heatmaps"
HEATMAP_DIR.mkdir(exist_ok=True)
JSON_DIR = config.OUT_DIR / "heatmap_json"
JSON_DIR.mkdir(exist_ok=True)

TOP_N = 20  # number of neighbors to visualize per concept


def fetch_concept_neighbors(conn, concept_name: str, slice_start: int, slice_end: int) -> Tuple[List[str], np.ndarray]:
    """Return neighbor tokens and their similarity to the canonical embedding."""
    with conn.cursor() as cur:
        # fetch canonical embedding
        cur.execute(
            "SELECT centroid FROM concept_slice_stats WHERE concept_name=%s AND slice_start=%s AND slice_end=%s;",
            (concept_name, slice_start, slice_end)
        )
        row = cur.fetchone()
        if not row:
            return [], np.array([])
        canonical_vec = np.array(row[0], dtype=np.float32)

        # fetch all tokens from this slice
        cur.execute(
            "SELECT token, vector FROM token_vectors WHERE slice_start=%s AND slice_end=%s;",
            (slice_start, slice_end)
        )
        rows = cur.fetchall()
        tokens = [r[0] for r in rows]
        vectors = [np.array(r[1], dtype=np.float32) for r in rows]

        # compute cosine similarity
        sims = np.array([1 - cosine(canonical_vec, v) for v in vectors], dtype=np.float32)

        # sort descending and pick top N
        idxs = np.argsort(-sims)[:TOP_N]
        top_tokens = [tokens[i] for i in idxs]
        top_sims = np.array([sims[i] for i in idxs], dtype=np.float32)
        return top_tokens, top_sims[:, None]


def save_json(concept_name: str, slice_range: Tuple[int,int], tokens: List[str], sims: np.ndarray):
    """Save a JSON dump of top neighbors and their similarity."""
    if not tokens:
        return

    data: Dict[str, List[List]] = {concept_name: []}
    for token, sim in zip(tokens, sims.flatten(), strict=True):
        data[concept_name].append([token, float(sim)])

    json_file = JSON_DIR / f"{concept_name}_{slice_range[0]}_{slice_range[1]}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {json_file}")


def plot_heatmap(concept_name: str, tokens: List[str], sims: np.ndarray, slice_range: Tuple[int,int]) -> None:
    """Plot a heatmap for a single concept in a given slice."""
    if sims.size == 0:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(tokens)*0.5), max(4, len(tokens)*0.5)))
    sns.heatmap(
        sims,
        annot=np.array(tokens)[:, None],
        fmt="",
        cmap="coolwarm",
        cbar=True,
        xticklabels=[concept_name],
        yticklabels=tokens,
        linewidths=0.5,
        linecolor="gray",
        square=False,
        ax=ax,
        vmin=0.8,
        vmax=1.0
    )
    ax.set_title(f"{concept_name} neighbors, slice {slice_range[0]}-{slice_range[1]}")
    plt.tight_layout()

    out_file = HEATMAP_DIR / f"{concept_name}_{slice_range[0]}_{slice_range[1]}.png"
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {out_file}")


def main():
    logger.info("Starting concept neighbor heatmap + JSON generation")

    with eebo_db.get_connection() as conn:
        for concept_name in config.CONCEPT_SETS.keys():
            for slice_range in config.SLICES:
                start, end = slice_range
                tokens, sims = fetch_concept_neighbors(conn, concept_name, start, end)
                if tokens:
                    plot_heatmap(concept_name, tokens, sims, slice_range)
                    save_json(concept_name, slice_range, tokens, sims)

    logger.info("Concept heatmap + JSON generation complete")


if __name__ == "__main__":
    main()
