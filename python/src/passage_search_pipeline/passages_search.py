#!/usr/bin/env python
"""
passage_search_pipeline/passages_search.py - search MacBERTh passage embeddings stored in LanceDB

Queries one or more year-bucketed Lance tables using a supplied query vector,
ranks results by cosine distance, and returns passage identifiers, corpus
metadata, and token offsets. Passage text and bibliographic metadata can be
retrieved separately from Postgres.

The module supports both text-query search and representation-based retrieval.
A query vector may be generated from text using `encode_passage_query` or
obtained from a previously retrieved passage's stored embedding for iterative
diachronic search.

Vectors are excluded from results by default. Set `include_vectors=True` in
`search_passages` when a caller needs the stored vectors for subsequent
retrieval steps.

Searches are restricted to the requested publication-year range. Results from
available tables are merged and ranked globally by cosine distance, with
smaller distances indicating greater similarity. Because each table contributes
at most `limit` candidates, the merged results are not guaranteed to be the
exact global nearest neighbours when approximate search is used.

The module provides the retrieval primitives for reverse-diachronic search;
it does not manage traversal between periods, deduplicate discovery paths,
or persist retrieval provenance.

The command-line interface accepts a text query, year range, representation,
and result limit, then prints ranked passages with bibliographic metadata.

Requires LanceDB tables written with compatible model, representation, vector
dimensions, and year-bucket configuration, plus access to the Postgres corpus
database for passage text and document metadata.
"""

from __future__ import annotations

import argparse

import lancedb

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_BUCKET_SIZE,
    LANCE_MODEL_NAME,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.macberth import PASSAGE_REPRESENTATIONS, encode_passage_query
from tier1.vector_writer import year_bucket


REPRESENTATIONS = PASSAGE_REPRESENTATIONS


def table_name(representation: str, year: int) -> str:
    start, end = year_bucket(year, LANCE_BUCKET_SIZE)
    return (
        f"passage_{representation}__{LANCE_MODEL_NAME}"
        f"__{start:04d}_{end:04d}"
    )



def candidate_tables(db, representation, start_year, end_year):
    available = set(db.list_tables().tables)
    names = set()

    for year in range(start_year, end_year + 1):
        names.add(table_name(representation, year))

    return sorted(names & available)


# Add near the top with the other imports
import numpy as np
from typing import Optional, List

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Both vectors are expected to be L2-normalised."""
    return float(np.dot(a, b))


def _l2_normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-magnitude vector")
    return v / n


def search_passages(
    db,
    query_vector,
    representation,
    start_year,
    end_year,
    limit,
    include_vectors: bool = False,
    negative_vectors: Optional[List[np.ndarray]] = None,
    beta: float = 0.5,
):
    """
    Search + optional negative-centroid re-ranking.

    If negative_vectors is supplied, vectors are always retrieved and
    results are re-ranked by:

        score = sim(query, doc) - beta * sim(neg_centroid, doc)

    Higher score is better. The original cosine distance is kept in the
    result dict for inspection.
    """
    # Force vectors when we need to re-rank
    need_vectors = include_vectors or (negative_vectors is not None)

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
    if need_vectors:
        columns.append("vector")

    per_table_limit = limit

    for name in candidate_tables(db, representation, start_year, end_year):
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

            if need_vectors and "vector" in row:
                # Ensure we have a clean float32 numpy vector
                row["vector"] = np.asarray(row["vector"], dtype=np.float32)

            results.append(row)

    if not results:
        return []

    # ------------------------------------------------------------------
    # Optional negative-centroid re-ranking
    # ------------------------------------------------------------------
    if negative_vectors:
        neg = np.stack([np.asarray(v, dtype=np.float32) for v in negative_vectors])
        neg_centroid = neg.mean(axis=0)
        norm = np.linalg.norm(neg_centroid)
        if norm < 1e-12:
            raise ValueError("Negative centroid has zero magnitude")
        neg_centroid = neg_centroid / norm

        neg = np.stack([_l2_normalise(v) for v in negative_vectors])
        neg_centroid = _l2_normalise(neg.mean(axis=0))

        query_vec = _l2_normalise(query_vector)

        for row in results:
            doc_vec = _l2_normalise(row["vector"])

            # Lance cosine distance is normally 1 − cos for unit vectors
            sim_q = 1.0 - row["distance"]
            # (or, more robustly: sim_q = float(np.dot(query_vec, doc_vec)))

            sim_neg = float(np.dot(doc_vec, neg_centroid))
            row["score"] = sim_q - beta * sim_neg
            row["sim_q"] = sim_q
            row["sim_neg"] = sim_neg

        results.sort(key=lambda r: r["score"], reverse=True)

    else:
        # Original behaviour
        results.sort(key=lambda r: r["distance"])

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
            JOIN tokens t ON t.corpus = d.corpus AND t.doc_id = d.doc_id
            WHERE d.corpus = %s
              AND d.doc_id = %s
              AND t.token_idx >= %s
              AND t.token_idx < %s
            GROUP BY d.title, d.author, d.filepath
            """,
            (
                result["corpus"],
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
    parser.add_argument( "--representation", choices=REPRESENTATIONS, default="mean4", )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    if args.limit < 1:
        parser.error("--limit must be positive")

    query_vector = encode_passage_query(
        args.query,
        args.representation,
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
