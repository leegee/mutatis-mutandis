
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR, OUT_DIR
from lib.corpus_db import get_connection
from lib.macberth import load_macberth, encode_passage_query
from tier1.passages_search import (
    REPRESENTATIONS,
    fetch_passage_text,
    search_passages,
)


DEFAULT_QUERIES = [
    "the rights and liberties of the people",
    "the lawful limits of royal power",
    "the people's consent to government",
]


def main() -> None:
    parser = argparse.ArgumentParser( description="Benchmark MacBERTh passage representations across different political queries." )
    parser.add_argument( "--query", action="append", dest="queries", help="Query to test. Repeat this option to supply several. " "Defaults to three diagnostic queries.", )
    parser.add_argument("--start-year", type=int, default=1600)
    parser.add_argument("--end-year", type=int, default=1650)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--output",
        default=OUT_DIR / "passage_vocabulary_benchmark.csv",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    if args.limit < 1:
        parser.error("--limit must be positive")

    queries = args.queries or DEFAULT_QUERIES

    print("Queries:")
    for i, query in enumerate(queries, start=1):
        print(f"  {i}. {query}")

    print("\nLoading MacBERTh...")
    macberth = load_macberth()
    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    all_rows = []
    text_cache = {}

    with get_connection(
        application_name="passage-vocabulary-benchmark"
    ) as conn:
        for query_number, query in enumerate(queries, start=1):
            for representation in REPRESENTATIONS:
                print(
                    f"\nQuery {query_number}/{len(queries)} | "
                    f"{representation}: {query}"
                )

                query_vector = encode_passage_query(
                    query,
                    representation,
                    macberth,
                )

                results = search_passages(
                    db,
                    query_vector,
                    representation,
                    args.start_year,
                    args.end_year,
                    args.limit,
                )

                for rank, result in enumerate(results, start=1):
                    cache_key = (
                        result["doc_id"],
                        result["token_start"],
                        result["token_end"],
                    )

                    if cache_key not in text_cache:
                        text_cache[cache_key] = fetch_passage_text(
                            conn, result
                        )

                    metadata = text_cache[cache_key]

                    all_rows.append({
                        "query_number": query_number,
                        "query": query,
                        "representation": representation,
                        "rank": rank,
                        "distance": result["distance"],
                        "passage_id": result["passage_id"],
                        "corpus": result["corpus"],
                        "doc_id": result["doc_id"],
                        "pub_year": result["pub_year"],
                        "title": metadata["title"],
                        "author": metadata["author"],
                        "word_start": result["word_start"],
                        "word_end": result["word_end"],
                        "token_start": result["token_start"],
                        "token_end": result["token_end"],
                        "filepath": metadata["filepath"],
                        "passage_text": metadata["passage_text"],
                        "table": result["table"],
                    })

                print(f"Retrieved {len(results)} passages.")
                for rank, result in enumerate(results[:5], start=1):
                    cache_key = (
                        result["doc_id"],
                        result["token_start"],
                        result["token_end"],
                    )
                    metadata = text_cache[cache_key]

                    print(
                        f"  {rank:2}. distance={result['distance']:.4f} "
                        f"year={result['pub_year']} "
                        f"doc_id={result['doc_id']}"
                    )
                    print(
                        "      "
                        + (metadata["passage_text"] or "[Text unavailable]")
                    )

    fields = [
        "query_number",
        "query",
        "representation",
        "rank",
        "distance",
        "passage_id",
        "corpus",
        "doc_id",
        "pub_year",
        "title",
        "author",
        "word_start",
        "word_end",
        "token_start",
        "token_end",
        "filepath",
        "passage_text",
        "table",
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved {len(all_rows)} rows to: {output_path.resolve()}")

    print("\nUnique passage overlap within each query:")
    for query_number, query in enumerate(queries, start=1):
        sets = {}
        for representation in REPRESENTATIONS:
            sets[representation] = {
                row["passage_id"]
                for row in all_rows
                if row["query_number"] == query_number
                and row["representation"] == representation
            }

        print(f"\n  {query}")
        for i, rep_a in enumerate(REPRESENTATIONS):
            for rep_b in REPRESENTATIONS[i + 1:]:
                shared = sets[rep_a] & sets[rep_b]
                print(
                    f"    {rep_a} vs {rep_b}: "
                    f"{len(shared)} shared passages"
                )


if __name__ == "__main__":
    main()