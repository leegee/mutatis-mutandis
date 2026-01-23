#!/usr/bin/env python
"""
build_role_profiles_per_slice.py

Phase 6: Build canonical role profiles per corpus slice.

- Reads canonical->neighbour neighbours with roles (from Phase 5)
- Aggregates role counts per canonical per slice
- Persists results to database for visualization or further analysis
"""

from __future__ import annotations
from typing import Dict, Tuple
from collections import defaultdict

from lib import eebo_db
from lib.eebo_logging import logger


SLICE_SIZE = 50000  # tokens per slice, can adjust as needed


def main(dry: bool = False) -> None:
    logger.info("Starting role profile aggregation per slice")

    # Role counts per canonical per slice: {(slice_start, slice_end, canonical, role): count}
    slice_role_counts: Dict[Tuple[int, int, str, str], int] = defaultdict(int)

    with eebo_db.get_connection(application_name="role_profiles") as conn:
        cur = conn.cursor()

        logger.info("Fetching canonical–neighbour–role–token data")
        cur.execute(
            """
            SELECT t.doc_id, t.token_idx, s.canonical, s.variant, s.role
            FROM tokens t
            JOIN spelling_map s ON s.variant = t.token
            WHERE s.role IS NOT NULL;
            """
        )

        for _doc_id, token_idx, canonical, _variant, role in cur:
            # Compute slice boundaries
            slice_start = (token_idx // SLICE_SIZE) * SLICE_SIZE
            slice_end = slice_start + SLICE_SIZE - 1
            key = (slice_start, slice_end, canonical, role)
            slice_role_counts[key] += 1

        logger.info(
            "Aggregated role counts for %d canonical/slice combinations",
            len(slice_role_counts)
        )

        if dry:
            logger.info("[DRY RUN] Role profiles not persisted to DB")
            _log_summary(slice_role_counts)
            return

        logger.info("Creating role_profiles table if not exists")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS role_profiles (
                slice_start INTEGER NOT NULL,
                slice_end INTEGER NOT NULL,
                canonical TEXT NOT NULL,
                role TEXT NOT NULL,
                count INTEGER NOT NULL,
                PRIMARY KEY (slice_start, slice_end, canonical, role)
            );
            """
        )

        logger.info("Inserting aggregated role counts into role_profiles table")
        rows_to_insert = [
            (slice_start, slice_end, canonical, role, count)
            for (slice_start, slice_end, canonical, role), count in slice_role_counts.items()
        ]

        cur.executemany(
            """
            INSERT INTO role_profiles (slice_start, slice_end, canonical, role, count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (slice_start, slice_end, canonical, role)
            DO UPDATE SET count = EXCLUDED.count;
            """,
            rows_to_insert
        )

        conn.commit()
        logger.info("Role profiles committed to DB")

    _log_summary(slice_role_counts)
    logger.info("Phase 6: Role profile aggregation complete")


def _log_summary(slice_role_counts: Dict[Tuple[int, int, str, str], int]) -> None:
    logger.info("Role profile summary:")
    summary: Dict[str, int] = defaultdict(int)
    for (_, _, _, role), count in slice_role_counts.items():
        summary[role] += count

    total = sum(summary.values())
    for role, count in sorted(summary.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total else 0.0
        logger.info("  %-15s %6d (%5.1f%%)", role, count, pct)


if __name__ == "__main__":
    main()
