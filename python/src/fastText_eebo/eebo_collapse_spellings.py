#!/usr/bin/env python3
"""
Collapse orthographic variants of neighbours at analysis time only.

Assumes an existing SQLite table:

neighbourhoods(
    slice_start INTEGER,
    slice_end INTEGER,
    query TEXT,
    neighbour TEXT,
    similarity REAL,
    rank INTEGER
)

Uses a spelling_map table to collapse variants to canonical forms.
"""

import sqlite3
import sys

import eebo_config as config
from eebo_db import dbh


QUERY_WORD = "liberty"
TOP_K_RAW = 50      # how deep to look before collapsing
TOP_K_CONCEPTS = 10 # how many concepts to report per slice


def ensure_spelling_map(conn):
    """
    Create spelling_map table if it does not exist.
    This is analysis-only and safe to re-run.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spelling_map (
            variant TEXT PRIMARY KEY,
            canonical TEXT NOT NULL
        )
    """)
    conn.commit()


def populate_minimal_spelling_map(conn):
    """
    Insert a conservative starter mapping.
    Extend this manually over time.
    """
    mappings = [
        ("liberty", "liberty"),
        ("libertie", "liberty"),
        ("libertye", "liberty"),
        ("libertyes", "liberty"),
    ]

    conn.executemany("""
        INSERT OR IGNORE INTO spelling_map (variant, canonical)
        VALUES (?, ?)
    """, mappings)
    conn.commit()


def fetch_collapsed_neighbours(conn):
    """
    Collapse neighbours to canonical forms and re-rank per slice.
    Uses mean similarity per concept.
    """
    query = f"""
    WITH collapsed AS (
        SELECT
            n.slice_start,
            n.slice_end,
            COALESCE(m.canonical, n.neighbour) AS concept,
            AVG(n.similarity) AS mean_similarity,
            COUNT(*) AS variant_count
        FROM neighbourhoods n
        LEFT JOIN spelling_map m
            ON n.neighbour = m.variant
        WHERE n.query = ?
          AND n.rank <= ?
        GROUP BY n.slice_start, n.slice_end, concept
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY slice_start
                ORDER BY mean_similarity DESC
            ) AS new_rank
        FROM collapsed
    )
    SELECT
        slice_start,
        slice_end,
        new_rank,
        concept,
        mean_similarity,
        variant_count
    FROM ranked
    WHERE new_rank <= ?
    ORDER BY slice_start, new_rank;
    """

    return conn.execute(
        query,
        (QUERY_WORD, TOP_K_RAW, TOP_K_CONCEPTS)
    ).fetchall()


def print_report(rows):
    current_slice = None

    for row in rows:
        slice_start, slice_end, rank, concept, sim, count = row

        if slice_start != current_slice:
            print()
            print(f"{slice_start}–{slice_end}")
            print("-" * 40)
            current_slice = slice_start

        print(
            f"{rank:>2}. {concept:<20} "
            f"sim={sim:.4f}  variants={count}"
        )


def main():
    conn = get_dbh()

    ensure_spelling_map(conn)
    populate_minimal_spelling_map(conn)

    rows = fetch_collapsed_neighbours(conn)

    if not rows:
        print("[INFO] No results found.")
        return

    print(f"[INFO] Collapsed neighbour report for '{QUERY_WORD}'")
    print_report(rows)


if __name__ == "__main__":
    main()
