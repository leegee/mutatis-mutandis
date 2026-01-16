#!/usr/bin/env python

"""
Collapse orthographic variants of neighbours at analysis time only.

Assumes an existing table:

neighbourhoods(
    slice_start INTEGER,
    slice_end INTEGER,
    query TEXT,
    neighbour TEXT,
    cosine DOUBLE PRECISION,
    rank INTEGER
)

Uses a spelling_map table to collapse variants to canonical forms.
"""

from collections import Counter
from typing import Counter as CounterType

import psycopg

import eebo_db
import eebo_config as config


QUERY_WORD = "liberty"
TOP_K_RAW = 75
TOP_K_CONCEPTS = 20


def populate_minimal_spelling_map(conn: psycopg.Connection) -> None:
    """
    Conservative, discipline-enforcing mappings.
    """

    mappings = [

        # liberty (orthographic only)
        ("liberty", "liberty", "orthographic"),
        ("libertie", "liberty", "orthographic"),
        ("libertye", "liberty", "orthographic"),
        ("libertyes", "liberty", "orthographic"),
        ("liberts", "liberty", "orthographic"),
        ("iberty", "liberty", "orthographic"),
        ("llberty", "liberty", "orthographic"),
        ("iiberty", "liberty", "orthographic"),
        ("lioerty", "liberty", "orthographic"),
        ("leberty", "liberty", "orthographic"),
        ("libertty", "liberty", "orthographic"),
        ("liberry", "liberty", "orthographic"),
        ("libertv", "liberty", "orthographic"),
        ("libertiu", "liberty", "orthographic"),
        ("libertly", "liberty", "orthographic"),

        # libertine family (derivational)
        ("libertine", "libertine", "derivational"),
        ("libertines", "libertine", "derivational"),
        ("libertin", "libertine", "derivational"),
        ("libertins", "libertine", "derivational"),
        ("libertinisme", "libertinism", "derivational"),
        ("libertinism", "libertinism", "derivational"),
        ("libertinous", "libertinous", "derivational"),
        ("libertinage", "libertinage", "derivational"),

        # freedom (orthographic only)
        ("freedom", "freedom", "orthographic"),
        ("freedome", "freedom", "orthographic"),
        ("freedoms", "freedom", "orthographic"),
        ("freedomes", "freedom", "orthographic"),
        ("freedoome", "freedom", "orthographic"),
        ("freedoom", "freedom", "orthographic"),
        ("freede", "freedom", "orthographic"),
        ("freeedome", "freedom", "orthographic"),

        # explicit exclusions
        ("philibert", "", "exclude"),
        ("sigibert", "", "exclude"),
        ("filibert", "", "exclude"),
        ("phylibert", "", "exclude"),
        ("macberth", "", "exclude"),
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO spelling_map (variant, canonical, concept_type)
            VALUES (%s, %s, %s)
            ON CONFLICT (variant) DO NOTHING
            """,
            mappings,
        )

    conn.commit()


def fetch_collapsed_neighbours(conn: psycopg.Connection):
    query = """
    WITH mapped AS (
        SELECT
            n.slice_start,
            n.slice_end,
            n.neighbour,
            n.cosine,
            n.query,
            m.canonical,
            m.concept_type
        FROM neighbourhoods n
        LEFT JOIN spelling_map m
            ON n.neighbour = m.variant
        WHERE n.query = %s
          AND n.rank <= %s
          AND COALESCE(m.concept_type, 'orthographic') != 'exclude'
    ),
    collapsed AS (
        SELECT
            slice_start,
            slice_end,
            COALESCE(
                canonical,
                CASE
                    WHEN neighbour LIKE query || '%%'
                     AND LENGTH(neighbour)
                         BETWEEN LENGTH(query)
                             AND LENGTH(query) + 3
                    THEN query
                    ELSE neighbour
                END
            ) AS concept,
            AVG(cosine) AS mean_similarity,
            MAX(cosine) AS peak_similarity,
            COUNT(*) AS variant_count
        FROM mapped
        GROUP BY slice_start, slice_end, concept
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY slice_start
                ORDER BY peak_similarity DESC, mean_similarity DESC
            ) AS new_rank
        FROM collapsed
    )
    SELECT
        slice_start,
        slice_end,
        new_rank,
        concept,
        mean_similarity,
        peak_similarity,
        variant_count
    FROM ranked
    WHERE new_rank <= %s
    ORDER BY slice_start, new_rank;
    """

    with conn.cursor() as cur:
        cur.execute(
            query,
            (QUERY_WORD, TOP_K_RAW, TOP_K_CONCEPTS),
        )
        return cur.fetchall()


def print_report(rows) -> None:
    current_slice = None

    for slice_start, slice_end, rank, concept, mean_sim, peak_sim, count in rows:
        if slice_start != current_slice:
            print()
            print(f"{slice_start}–{slice_end}")
            print("-" * 40)
            current_slice = slice_start

        print(
            f"{rank:>2}. {concept:<18} "
            f"peak={peak_sim:.4f} mean={mean_sim:.4f} variants={count}"
        )


def fetch_absorption_report(conn: psycopg.Connection):
    query = """
    SELECT
        m.variant,
        m.canonical,
        m.concept_type,
        COUNT(*) AS occurrences,
        AVG(n.cosine) AS mean_similarity,
        MAX(n.cosine) AS peak_similarity,
        MIN(n.slice_start) AS first_seen,
        MAX(n.slice_end) AS last_seen
    FROM neighbourhoods n
    JOIN spelling_map m
        ON n.neighbour = m.variant
    WHERE n.query = %s
      AND m.concept_type != 'exclude'
    GROUP BY m.variant, m.canonical, m.concept_type
    ORDER BY
        m.concept_type,
        mean_similarity DESC,
        occurrences DESC;
    """

    with conn.cursor() as cur:
        cur.execute(query, (QUERY_WORD,))
        return cur.fetchall()


def print_absorption_report(rows) -> None:
    print()
    print("Absorbed spellings (classified)")
    print("=" * 60)

    for (
        variant,
        canonical,
        ctype,
        occ,
        mean_sim,
        peak_sim,
        first_seen,
        last_seen,
    ) in rows:

        absorption_class = classify_absorption(mean_sim, occ)

        print(
            f"{variant:<15} → {canonical:<12} "
            f"[{ctype:<12}] "
            f"{absorption_class:<8} "
            f"mean={mean_sim:.4f} peak={peak_sim:.4f} "
            f"slices={occ} ({first_seen}–{last_seen})"
        )


def classify_absorption(mean_similarity: float, slice_count: int) -> str:
    if (
        mean_similarity >= config.STRONG_MEAN
        and slice_count >= config.MIN_SLICES_STRONG
    ):
        return "strong"
    elif mean_similarity >= config.MODERATE_MEAN:
        return "moderate"
    else:
        return "weak"


def summarise_absorption(rows) -> None:
    counter: CounterType[str] = Counter()

    for (_, _, _, occ, mean_sim, *_ ) in rows:
        cls = classify_absorption(mean_sim, occ)
        counter[cls] += 1

    print()
    print("Absorption class summary")
    print("-" * 40)
    for cls in ("strong", "moderate", "weak"):
        print(f"{cls:<8}: {counter.get(cls, 0)}")


def main() -> None:
    conn = eebo_db.dbh

    populate_minimal_spelling_map(conn)

    rows = fetch_collapsed_neighbours(conn)

    if not rows:
        print("[INFO] No results found.")
        return

    print(f"[INFO] Collapsed neighbour report for '{QUERY_WORD}'")
    print_report(rows)

    absorption = fetch_absorption_report(conn)
    if absorption:
        print_absorption_report(absorption)
        summarise_absorption(absorption)


if __name__ == "__main__":
    main()
