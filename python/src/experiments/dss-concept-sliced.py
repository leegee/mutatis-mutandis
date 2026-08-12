#!/usr/bin/env python
"""
experiments/dss-concept-sliced.py

Observation-level semantic backcasting.

Given occurrences of a concept in a recent period,
find earlier contextual observations occupying similar
semantic positions.

No centroids.
No clustering.
No dimensionality reduction.
"""

from __future__ import annotations

import argparse
import json
import sqlite3

from lib.corpus_config import (
    CORPUS_TIER2_DB_PATH,
    CORPUS_MAX_YEAR,
    EVENTSTORE_T1_PATH,
    TMP_DIR
)

from lib.corpus_faiss import CorpusFaissIndex
from lib.corpus_logging import logger
from lib.zarr_event_lookup import ZarrEventLookup


def sqlite_connection():
    con = sqlite3.connect(CORPUS_TIER2_DB_PATH)
    con.execute("PRAGMA busy_timeout=5000")
    return con


def load_field_events(con, concept, start_year, end_year):
    return con.execute(
        """
        SELECT
            e.event_id,
            e.pub_year
        FROM concept_field_events f
        JOIN events e
          ON e.event_id = f.event_id
        WHERE f.concept = ?
          AND e.pub_year BETWEEN ? AND ?
        ORDER BY e.pub_year, e.event_id
        """,
        (
            concept,
            start_year,
            end_year,
        ),
    ).fetchall()


def load_indexes(start_year, end_year):
    return CorpusFaissIndex.load_existing_range(
        start_year=start_year,
        end_year=end_year,
        workers=8,
    )


def historical_slices(start, end, width=10):
    year = start
    while year <= end:
        yield year, min(year + width - 1, end)
        year += width


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}

    for ranked in ranked_lists.values():
        for rank, (_, eid, year, scale) in enumerate(
            ranked,
            start=1,
        ):
            if eid not in scores:
                scores[eid] = {
                    "rrf_score": 0.0,
                    "year": year,
                    "scale": scale,
                }

            scores[eid]["rrf_score"] += (
                1.0 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1]["rrf_score"],
        reverse=True,
    )


def search_historical(historical_index, source_vectors, top_k):
    results = []
    scales = ("local", "medium", "broad")

    n = len(source_vectors["local"])

    for i in range(n):
        scale_results = {
            scale: []
            for scale in scales
        }

        for scale in scales:
            query = source_vectors[scale][i:i + 1]

            for year, indexes in historical_index.items():
                scores, ids = indexes[scale].search(
                    query,
                    top_k,
                )

                for score, eid in zip(
                    scores[0],
                    ids[0],
                ):
                    if eid == -1:
                        continue

                    scale_results[scale].append(
                        (
                            float(score),
                            int(eid),
                            year,
                            scale,
                        )
                    )

            scale_results[scale].sort(
                key=lambda x: x[0],
                reverse=True,
            )

        fused = reciprocal_rank_fusion(scale_results)
        results.append(fused[:top_k])

    return results


def combine_historical_tokens(matches, lookup):
    field = {}

    for ranked in matches:
        seen = set()

        for eid, data in ranked:
            if eid in seen:
                continue

            seen.add(eid)

            pos = lookup.get_pos(eid)
            token = str(
                lookup.token[pos]
            ).lower()

            if token not in field:
                field[token] = {
                    "weight": 0.0,
                    "events": 0,
                    "years": set(),
                }

            field[token]["weight"] += data["rrf_score"]
            field[token]["events"] += 1
            field[token]["years"].add(
                int(data["year"])
            )

    return sorted(
        field.items(),
        key=lambda x: x[1]["weight"],
        reverse=True,
    )


def serialise_field(field, limit=100):
    total = sum(
        data["weight"]
        for _, data in field
    )

    return [
        {
            "token": token,
            "weight": round(data["weight"], 6),
            "weight_norm": round(
                data["weight"] / total,
                8,
            ) if total else 0.0,
            "events": data["events"],
            "years": sorted(data["years"]),
        }
        for token, data in field[:limit]
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument( "--concept", default="REVOLUTION", )
    parser.add_argument( "--neighbours", type=int, default=10, )
    parser.add_argument( "--early_start", type=int, default=1630, )
    parser.add_argument( "--early_end", type=int, default=1800, )
    parser.add_argument( "--slice_width", type=int, default=10, )
    parser.add_argument( "--late_start", type=int, default=CORPUS_MAX_YEAR - 10, )
    parser.add_argument( "--late_end", type=int, default=CORPUS_MAX_YEAR, )
    parser.add_argument( "--output", default=TMP_DIR/"dss_semantic_trajectory.json", )
    args = parser.parse_args()

    logger.info( f"[backcast] source={args.late_start}-{args.late_end}" )

    con = sqlite_connection()

    source_events = load_field_events(
        con,
        args.concept,
        args.late_start,
        args.late_end,
    )

    event_ids = [
        int(row[0])
        for row in source_events
    ]

    logger.info( f"[backcast] source observations={len(event_ids)}" )

    source_index = CorpusFaissIndex.load_all(
        workers=8,
    )

    lookup = ZarrEventLookup(
        EVENTSTORE_T1_PATH
    )

    lookup.attach_index(
        source_index
    )

    positions = [
        lookup.get_pos(eid)
        for eid in event_ids
    ]

    source_vectors = {
        "local": lookup.emb_local[positions],
        "medium": lookup.emb_medium[positions],
        "broad": lookup.emb_broad[positions],
    }

    trajectory = {}

    for start, end in historical_slices(
        args.early_start,
        args.early_end,
        args.slice_width,
    ):
        logger.info( f"[backcast] comparing={start}-{end}" )

        try:
            historical_index = load_indexes(
                start,
                end,
            )
        except RuntimeError:
            logger.info( f"[backcast] skipping empty slice={start}-{end}" )
            continue

        matches = search_historical(
            historical_index,
            source_vectors,
            args.neighbours,
        )

        field = combine_historical_tokens(
            matches,
            lookup,
        )

        trajectory[f"{start}-{end}"] = serialise_field(field)

    with open(
        args.output,
        "w",
        encoding="utf8",
    ) as f:
        json.dump(
            {
                "concept": args.concept,
                "source": f"{args.late_start}-{args.late_end}",
                "trajectory": trajectory,
            },
            f,
            indent=2,
        )

    con.close()


if __name__ == "__main__":
    main()