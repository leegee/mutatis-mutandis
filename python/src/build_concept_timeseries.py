#!/usr/bin/env python
"""
build_concept_timeseries.py

For each concept and each temporal slice:
- Gather vectors for all variant forms
- Compute centroid + variance
- Store slice-level concept statistics
"""

from __future__ import annotations
import numpy as np
from typing import Iterable, List

import lib.eebo_config as config
import lib.eebo_db as eebo_db
from lib.eebo_logging import logger


def fetch_vectors_for_forms(conn, forms: Iterable[str], start: int, end: int):
    """Fetch vectors for a set of forms in a given slice."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT token, vector
            FROM token_vectors
            WHERE slice_start = %s
              AND slice_end   = %s
              AND token = ANY(%s);
            """,
            (start, end, list(forms)),
        )
        rows = cur.fetchall()

    tokens = [r[0] for r in rows]
    vectors = [np.array(r[1], dtype=np.float32) for r in rows]

    return tokens, vectors


def compute_centroid_and_variance(vectors: List[np.ndarray]):
    """Return centroid and mean squared distance from centroid."""
    mat = np.vstack(vectors)
    centroid = mat.mean(axis=0)

    diffs = mat - centroid
    variance = float(np.mean(np.sum(diffs * diffs, axis=1)))

    return centroid.astype(np.float32), variance


def store_concept_stats(conn, concept_name, start, end, centroid, variance, tokens):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO concept_slice_stats
                (concept_name, slice_start, slice_end, centroid, variance, token_count, forms_used)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (concept_name, slice_start, slice_end)
            DO UPDATE SET
                centroid    = EXCLUDED.centroid,
                variance    = EXCLUDED.variance,
                token_count = EXCLUDED.token_count,
                forms_used  = EXCLUDED.forms_used;
            """,
            (
                concept_name,
                start,
                end,
                centroid.tolist(),
                variance,
                len(tokens),
                tokens,
            ),
        )


def main():
    logger.info("Starting concept time-series construction")

    with eebo_db.get_connection() as conn:
        for concept_name, cfg in config.CONCEPT_SETS.items():
            forms = cfg["forms"]
            false_pos = set(cfg.get("false_positives", []))

            logger.info(f"Processing concept: {concept_name}")

            for start, end in config.SLICES:
                tokens, vectors = fetch_vectors_for_forms(conn, forms, start, end)

                # remove false positives
                filtered = [
                    (t, v)
                    for t, v in zip(tokens, vectors, strict=True)
                    if t not in false_pos
                ]

                if not filtered:
                    continue

                tokens, vectors = zip(*filtered, strict=True)

                centroid, variance = compute_centroid_and_variance(list(vectors))

                store_concept_stats(
                    conn,
                    concept_name,
                    start,
                    end,
                    centroid,
                    variance,
                    list(tokens),
                )

        conn.commit()

    logger.info("Concept time-series complete")


if __name__ == "__main__":
    main()
