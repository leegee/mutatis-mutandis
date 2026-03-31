#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from mb_embedding_pipeline import load_vectors
from lib.eebo_config import CONCEPT_SETS
from lib.eebo_logging import logger


SLICES = [
    (1625, 1629),
    (1630, 1634),
    (1635, 1639),
    (1640, 1640),
    (1641, 1641),
    (1642, 1642),
    (1643, 1643),
    (1644, 1644),
    (1645, 1645),
    (1646, 1646),
    (1647, 1647),
    (1648, 1648),
    (1649, 1649),
    (1650, 1650),
    (1651, 1651),
    (1652, 1654),
    (1655, 1657),
]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_centroid(slice_id, token):
    data = load_vectors(slice_id)
    vecs = data.get(token, [])

    if not vecs:
        logger.debug(f"[{slice_id}] token='{token}' → no occurrences")
        return None

    logger.debug(f"[{slice_id}] token='{token}' → n={len(vecs)}")

    return np.mean(np.stack(vecs), axis=0)


def compute_drift_series(token):
    centroids = []
    slice_years = []

    for start, end in SLICES:
        sid = f"{start}-{end}"
        c = compute_centroid(sid, token)

        if c is not None:
            centroids.append(c)
            slice_years.append(start)

    if len(centroids) < 2:
        logger.warning(f"token='{token}' insufficient data for drift (n={len(centroids)})")
        return [], []

    drifts = []
    drift_x = []

    for i in range(1, len(centroids)):
        d = 1 - cosine(centroids[i], centroids[i - 1])
        drifts.append(d)
        drift_x.append(slice_years[i])

        logger.debug(
            f"token='{token}' drift {slice_years[i-1]}→{slice_years[i]} = {d:.4f}"
        )

    logger.info(
        f"token='{token}' computed drift series (points={len(drifts)})"
    )

    return drift_x, drifts


def main():
    logger.info("Starting drift computation (canonical tokens only)")

    plt.figure(figsize=(12, 6))

    for concept in CONCEPT_SETS.keys():
        token = concept.lower()
        logger.info(f"Processing concept='{concept}' token='{token}'")

        x, y = compute_drift_series(token)

        if not y:
            logger.warning(f"Skipping concept='{concept}' (no drift data)")
            continue

        plt.plot(x, y, marker='o', label=concept)

    plt.xlabel("Year (start of slice)")
    plt.ylabel("Drift (1 - cosine)")
    plt.title("Per-slice Drift (Canonical Tokens Only)")
    plt.legend()
    plt.tight_layout()

    logger.info("Rendering plot")
    plt.show()

    logger.info("Done")


if __name__ == "__main__":
    main()
