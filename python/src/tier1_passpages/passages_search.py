# passages_search.py


from __future__ import annotations

import argparse
from collections import defaultdict

import lancedb
import numpy as np

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_BUCKET_SIZE,
    LANCE_MODEL_NAME,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import load_macberth, encode_passage_query
from tier1.vector_writer import year_bucket


REPRESENTATIONS = ("l8", "last", "mean4")


def table_name(representation: str, year: int) -> str:
    start, end = year_bucket(year, LANCE_BUCKET_SIZE)
    return (
        f"passage_{representation}__{LANCE_MODEL_NAME}"
        f"__{start:04d}_{end:04d}"
    )


def pick_layers(hidden_states):
    import torch
    return {
        "l8": hidden_states[8],
        "last": hidden_states[-1],
        "mean4": torch.stack(hidden_states[-4:]).mean(0),
    }


def validate_periods(periods):
    if not periods:
        raise ValueError("At least one period is required")

    for start, end in periods:
        if start > end:
            raise ValueError(f"Invalid period: {start}-{end}")

    for current, earlier in zip(periods, periods[1:]):
        if earlier[1] >= current[0]:
            raise ValueError(
                "Periods must be ordered backwards "
                "and must not overlap"
            )


def candidate_tables(db, representation, start_year, end_year):
    available = set(db.list_tables().tables)
    names = set()

    for year in range(start_year, end_year + 1):
        names.add(table_name(representation, year))

    return sorted(names & available)


def search_passages(
    db,
    query_vector,
    representation,
    start_year,
    end_year,
    limit,
    include_vectors=False,
):
    results = []

    columns = [
        "passage_id",
        "corpus",
        "doc_id",
        "pub_year",
        "word_start",
        "word_end",
        "token_start",
        "token_end",
        "_distance",
    ]
    if include_vectors:
        columns.append("vector")

    per_table_limit = limit

    for name in candidate_tables(
        db, representation, start_year, end_year
    ):
        table = db.open_table(name)

        arrow = (
            table.search(
                query_vector,
                vector_column_name="vector",
            )
            .metric("cosine")
            .where(
                f"pub_year >= {int(start_year)} "
                f"AND pub_year <= {int(end_year)}"
            )
            .select(columns)
            .limit(per_table_limit)
            .to_arrow()
        )

        for row in arrow.to_pylist():
            distance = row.get("_distance")
            if distance is None:
                raise RuntimeError(
                    "Lance search returned no _distance column. "
                    "Check the installed LanceDB API."
                )

            row["distance"] = float(distance)
            row["table"] = name
            results.append(row)

    results.sort(key=lambda row: row["distance"])
    return results[:limit]


def fetch_passage_text(conn, result):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                d.title,
                d.author,
                d.filepath,
                string_agg(
                    t.token,
                    ' ' ORDER BY t.token_idx
                ) AS passage_text
            FROM documents d
            JOIN tokens t ON t.doc_id = d.doc_id
            WHERE d.doc_id = %s
              AND t.token_idx >= %s
              AND t.token_idx < %s
            GROUP BY d.title, d.author, d.filepath
            """,
            (
                result["doc_id"],
                result["token_start"],
                result["token_end"],
            ),
        )
        row = cur.fetchone()

    if row is None:
        return {
            "title": None,
            "author": None,
            "filepath": None,
            "passage_text": None,
        }

    return {
        "title": row[0],
        "author": row[1],
        "filepath": row[2],
        "passage_text": row[3],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Search MacBERTh passage embeddings in Lance."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="mean4",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    if args.limit < 1:
        parser.error("--limit must be positive")

    macberth = load_macberth()

    query_vector = encode_passage_query(
        args.query,
        args.representation,
        macberth,
    )

    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    results = search_passages(
        db,
        query_vector,
        args.representation,
        args.start_year,
        args.end_year,
        args.limit,
    )

    if not results:
        print("No passages found for this date range.")
        return

    with get_connection(
        application_name="passage-search"
    ) as conn:
        for rank, result in enumerate(results, start=1):
            metadata = fetch_passage_text(conn, result)

            print()
            print(
                f"{rank}. year={result['pub_year']} "
                f"distance={result['distance']:.4f}"
            )
            print(
                f"   corpus={result['corpus']} "
                f"doc_id={result['doc_id']}"
            )
            print(
                f"   title={metadata['title'] or '(untitled)'}"
            )
            print(
                f"   words={result['word_start']}:"
                f"{result['word_end']} "
                f"tokens={result['token_start']}:"
                f"{result['token_end']}"
            )
            print(f"   filepath={metadata['filepath']}")
            print(f"   {metadata['passage_text'] or '[Text unavailable]'}")


if __name__ == "__main__":
    main()
