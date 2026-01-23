#!/usr/bin/env python
"""
annotate_neighbors_with_roles.py

Phase 5: Annotates canonical–neighbour pairs with conceptual roles.

- LEFT JOINs to catch missing vectors
- Logs missing vectors
- Assigns a role per neighbour where both vectors exist
- Updates spelling_map.role
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from lib import eebo_db
from lib.eebo_logging import logger


def assign_role(
    canonical: str,
    neighbour: str,
    c_vec: np.ndarray,
    n_vec: np.ndarray
) -> str:
    """Heuristic role assignment based on vector similarity / morphology."""
    cos = float(np.dot(c_vec, n_vec) / (np.linalg.norm(c_vec) * np.linalg.norm(n_vec)))
    delta = n_vec - c_vec
    d = float(np.dot(delta, c_vec) / np.dot(c_vec, c_vec))

    # Morphological signals
    if neighbour.startswith(("un", "in", "non", "anti")):
        return "antithetical"
    if neighbour.endswith(("ness", "ity", "ship")):
        return "normative"

    # Vector-based rules
    if cos > 0.8 and abs(d) < 0.05:
        return "synonymic"
    if d > 0.1:
        return "generalising"
    if d < -0.1:
        return "specifying"
    if cos < 0.6:
        return "metaphorical"

    return "noise"


def _log_role_summary(role_counts: Dict[str, int]) -> None:
    """Logs a summary of roles assigned."""
    logger.info("Role distribution summary:")
    total = sum(role_counts.values())
    for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total else 0.0
        logger.info("  %-15s %6d  (%5.1f%%)", role, count, pct)


def main(dry: bool = False) -> None:
    logger.info("Starting role annotation phase")

    role_counts: Dict[str, int] = {}
    canonical_counts: Dict[str, int] = {}

    with eebo_db.get_connection(application_name="annotate_roles") as conn:
        cur = conn.cursor()

        logger.info("Fetching canonical–neighbour vector pairs from DB")
        cur.execute(
            """
            SELECT
                s.canonical,
                s.variant,
                c.vector AS c_vec,
                t.vector AS n_vec
            FROM spelling_map s
            LEFT JOIN canonical_centroids c
                ON c.canonical = s.canonical
            LEFT JOIN token_vectors t
                ON t.token = s.variant
            """
        )

        rows = cur.fetchall()
        logger.info("Fetched %d rows from spelling_map", len(rows))

        valid_rows: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
        missing_c_vec = 0
        missing_n_vec = 0

        for canonical, neighbour, c_blob, n_blob in rows:
            if c_blob is None:
                missing_c_vec += 1
                continue
            if n_blob is None:
                missing_n_vec += 1
                continue
            valid_rows.append((canonical, neighbour, np.array(c_blob, dtype=np.float32), np.array(n_blob, dtype=np.float32)))

        logger.info(
            "Skipping %d rows with missing canonical vectors, %d rows with missing neighbour vectors",
            missing_c_vec,
            missing_n_vec
        )
        logger.info("Processing %d rows for role assignment", len(valid_rows))

        updates: List[Tuple[str, str, str]] = []

        for canonical, neighbour, c_vec, n_vec in valid_rows:
            role = assign_role(canonical, neighbour, c_vec, n_vec)
            role_counts[role] = role_counts.get(role, 0) + 1
            canonical_counts[canonical] = canonical_counts.get(canonical, 0) + 1
            updates.append((role, canonical, neighbour))

        logger.info(
            "Computed %d role assignments across %d canonicals",
            len(updates),
            len(canonical_counts)
        )

        if dry:
            logger.info("[DRY RUN] No database updates performed")
            _log_role_summary(role_counts)
            return

        if updates:
            logger.info("Writing role annotations to spelling_map.role")
            cur.executemany(
                """
                UPDATE spelling_map
                SET role = %s
                WHERE canonical = %s AND variant = %s;
                """,
                updates
            )
            conn.commit()
            logger.info("Database update committed")

    _log_role_summary(role_counts)
    logger.info("Role annotation phase complete")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Dry run, no DB writes")
    args = parser.parse_args()

    main(dry=args.dry)
