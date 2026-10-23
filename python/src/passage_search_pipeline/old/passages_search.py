#!/usr/bin/env python
"""
passage_search_pipeline/passages_search.py

Search MacBERTh passage embeddings stored in year-bucketed LanceDB tables.

This single script is both a library (search_passages, load_passage_vector,
fetch_passage_text, ...) and a command-line tool. It replaces the former
passages_search.py + passages_vector_search.py pair.

Query vector sources (one is required):
  --query TEXT        encode text with MacBERTh
  --vector-id ID      reuse the stored vector of an existing passage
                      (find IDs with find_doc_vectors.py)

Optional negative re-ranking:
  --negative-id ID    passage to steer away from (repeatable)
  --beta B            weight of the negative centroid (default 0.5)
  --overfetch N       candidate multiplier per table when negatives are
                      used (default 5), so the penalty can pull in
                      passages that were not already in the top `limit`

    score = cos(query, doc) - beta * cos(negative_centroid, doc)

Examples:

    python src/passage_search_pipeline/passages_search.py \
        --query "white hair" --limit 30

    python src/passage_search_pipeline/passages_search.py \\
        --vector-id ef2f4aaf68dc6c924a2f1218e470a555060be2cb886a0ef1ebf169b800508f8a \
        --limit 30

    python src/passage_search_pipeline/passages_search.py \
        --query "white hair" --representation l8 --limit 50 \
        --negative-id dfbee39870fbbababa5fc24da247fb8468a82350bca1170209458b84568be2c3 \
        --negative-id afee38e3452766ffd28bc725ba733cec845fe8cb338c8ef93a52f850223fdbe7 \
        --negative-id b18fa10570b4402f240375c0f45fe9d179493486ab8abe32f285f112b7ff8126 \
        --beta 0.6

Notes:
  * Results from all tables in the year range are merged and ranked
    globally. Each table contributes at most `limit` candidates (or
    `limit * overfetch` with negatives), so with approximate search the
    merged list is not guaranteed to be the exact global top-k.
  * Passage text and bibliographic metadata come from Postgres.
  * This module provides retrieval primitives only; it does not manage
    traversal between periods or persist retrieval provenance.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import lancedb
import numpy as np

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_MODEL_NAME,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import get_logger
from lib.macberth import PASSAGE_REPRESENTATIONS, encode_passage_query
from tier1.vector_writer import year_bucket  # noqa: F401  (kept for callers)

logger = get_logger("passages_search")

REPRESENTATIONS = PASSAGE_REPRESENTATIONS

RESULT_COLUMNS = [
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


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------

def normalise_vector(vector, description: str = "vector") -> np.ndarray:
    """Return a finite, L2-normalised float32 copy of `vector`."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if not np.isfinite(vector).all():
        raise ValueError(f"{description} contains NaN or infinity")

    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        raise ValueError(f"{description} has zero magnitude")

    return vector / norm


# ---------------------------------------------------------------------------
# Table selection
# ---------------------------------------------------------------------------

def table_prefix(representation: str) -> str:
    return f"passage_{representation}__{LANCE_MODEL_NAME}__"


def candidate_tables(db, representation, start_year, end_year):
    """Names of existing tables whose year bucket overlaps the range."""
    prefix = table_prefix(representation)
    names = []

    for name in db.list_tables().tables:
        if not name.startswith(prefix):
            continue
        try:
            b_start, b_end = (int(x) for x in name[len(prefix):].split("_"))
        except ValueError:
            logger.warning("Skipping table with unparseable name: %s", name)
            continue
        if b_start <= end_year and b_end >= start_year:
            names.append(name)

    return sorted(names)


# ---------------------------------------------------------------------------
# Loading stored vectors
# ---------------------------------------------------------------------------

def load_passage_vector(db, passage_id, representation) -> np.ndarray:
    """Load an existing passage's stored vector (normalised)."""
    prefix = table_prefix(representation)
    escaped_id = passage_id.replace("'", "''")
    matches = []

    for table_name in db.list_tables().tables:
        if not table_name.startswith(prefix):
            continue

        table = db.open_table(table_name)
        rows = (
            table.search()
            .where(f"passage_id = '{escaped_id}'")
            .limit(2)
            .to_list()
        )
        for row in rows:
            matches.append((table_name, row))

    if not matches:
        raise ValueError(
            f"No vector found for passage_id={passage_id!r}, "
            f"representation={representation!r}"
        )

    if len(matches) > 1:
        locations = [name for name, _ in matches]
        raise ValueError(
            f"Multiple vectors found for passage_id={passage_id!r}: "
            f"{locations}"
        )

    table_name, row = matches[0]
    vector = normalise_vector(row["vector"], "Stored passage vector")

    logger.info(
        "Loaded vector ID %s from %s (doc_id=%s, year=%s)",
        passage_id,
        table_name,
        row.get("doc_id"),
        row.get("pub_year"),
    )
    return vector


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

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
    overfetch: int = 5,
):
    """
    Search year-bucketed tables, with optional negative-centroid re-ranking.

    Without negatives, results are ordered by cosine distance (ascending).

    With negatives, each table is queried for `limit * overfetch`
    candidates, which are re-ranked by

        score = sim(query, doc) - beta * sim(neg_centroid, doc)

    (higher is better) and truncated to `limit`. The original cosine
    distance is kept in each result as `distance`; `sim_q`, `sim_neg` and
    `score` are added.
    """
    query_vec = normalise_vector(query_vector, "Query vector")
    use_negatives = bool(negative_vectors)
    need_vectors = include_vectors or use_negatives

    columns = list(RESULT_COLUMNS)
    if need_vectors:
        columns.append("vector")

    per_table_limit = limit * max(1, overfetch) if use_negatives else limit

    results = []

    for name in candidate_tables(db, representation, start_year, end_year):
        table = db.open_table(name)

        arrow = (
            table.search(query_vec, vector_column_name="vector")
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
                row["vector"] = np.asarray(row["vector"], dtype=np.float32)
            results.append(row)

    if not results:
        return []

    if use_negatives:
        neg = np.stack(
            [normalise_vector(v, "Negative vector") for v in negative_vectors]
        )
        neg_centroid = normalise_vector(neg.mean(axis=0), "Negative centroid")

        for row in results:
            doc_vec = normalise_vector(row["vector"], "Document vector")
            sim_q = float(np.dot(query_vec, doc_vec))
            sim_neg = float(np.dot(neg_centroid, doc_vec))
            row["sim_q"] = sim_q
            row["sim_neg"] = sim_neg
            row["score"] = sim_q - beta * sim_neg

        results.sort(key=lambda r: r["score"], reverse=True)
    else:
        results.sort(key=lambda r: r["distance"])

    results = results[:limit]

    if not include_vectors:
        for row in results:
            row.pop("vector", None)

    return results


# ---------------------------------------------------------------------------
# Postgres lookup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Search historical passages using MacBERTh embeddings."
    )
    parser.add_argument(
        "--query",
        help="Text to encode as the search query.",
    )
    parser.add_argument(
        "--vector-id",
        help="Use an existing passage vector (by passage_id) as the query "
             "instead of encoding --query.",
    )
    parser.add_argument(
        "--start-year", type=int, default=0,
        help="Earliest publication year (default: 0).",
    )
    parser.add_argument(
        "--end-year", type=int, default=3000,
        help="Latest publication year (default: 3000).",
    )
    parser.add_argument(
        "--representation", choices=REPRESENTATIONS, default="l8",
        help="Passage representation (default: l8).",
    )
    parser.add_argument(
        "--limit", type=int, default=30,
        help="Maximum number of results (default: 30).",
    )
    parser.add_argument(
        "--negative-id", action="append", default=[],
        help="Passage ID of a negative example (repeatable). These are "
             "averaged into a centroid used for margin re-ranking.",
    )
    parser.add_argument(
        "--beta", type=float, default=0.5,
        help="Weight of the negative centroid in re-ranking "
             "(score = sim_q - beta * sim_neg). Default: 0.5",
    )
    parser.add_argument(
        "--overfetch", type=int, default=5,
        help="Per-table candidate multiplier when negatives are used "
             "(default: 5).",
    )
    args = parser.parse_args()

    if not args.query and not args.vector_id:
        parser.error("one of --query or --vector-id is required")
    if args.query and args.vector_id:
        logger.warning("Both --query and --vector-id given; using --vector-id")
    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.overfetch < 1:
        parser.error("--overfetch must be >= 1")

    return args


def log_result(rank, result, metadata):
    logger.info("...............................................................")
    if "score" in result:
        logger.info(
            f"{rank}. year={result['pub_year']} "
            f"distance={result['distance']:.4f} "
            f"score={result['score']:.4f} "
            f"sim_neg={result['sim_neg']:.4f}"
        )
    else:
        logger.info(
            f"{rank}. year={result['pub_year']} "
            f"distance={result['distance']:.4f}"
        )
    logger.info(f"   corpus={result['corpus']} doc_id={result['doc_id']}")
    logger.info(f"   title={metadata['title'] or '(untitled)'}")
    logger.info(f"   author={metadata['author'] or '(unknown)'}")
    logger.info(f"   filepath={metadata['filepath']}")
    logger.info(
        f"   words={result['word_start']}:{result['word_end']} "
        f"tokens={result['token_start']}:{result['token_end']}"
    )
    logger.info(f"   passage_id={result['passage_id']}")
    logger.info(f"   {metadata['passage_text'] or '[Text unavailable]'}")


def main():
    args = parse_args()

    logger.info("Lance directory: %s", LANCE_INDEXES_DIR)
    logger.info("Years: %s-%s", args.start_year, args.end_year)
    logger.info("Representation: %s", args.representation)
    logger.info("Limit: %s", args.limit)

    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    if args.vector_id:
        logger.info("Query vector ID: %s", args.vector_id)
        query_vector = load_passage_vector(
            db, args.vector_id, args.representation
        )
    else:
        logger.info("Text query: %s", args.query)
        query_vector = normalise_vector(
            encode_passage_query(args.query, args.representation),
            "Text query",
        )

    negative_vectors = []
    for nid in args.negative_id:
        try:
            negative_vectors.append(
                load_passage_vector(db, nid, args.representation)
            )
            logger.info("Loaded negative vector: %s", nid)
        except ValueError as e:
            logger.warning("Skipping negative %s: %s", nid, e)

    if args.negative_id and not negative_vectors:
        logger.warning(
            "No valid negative vectors loaded; proceeding without re-ranking"
        )

    results = search_passages(
        db=db,
        query_vector=query_vector,
        representation=args.representation,
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
        include_vectors=False,
        negative_vectors=negative_vectors or None,
        beta=args.beta,
        overfetch=args.overfetch,
    )

    if not results:
        logger.info("No results found.")
        return

    with get_connection(application_name="passage-search") as conn:
        for rank, result in enumerate(results, start=1):
            metadata = fetch_passage_text(conn, result)
            log_result(rank, result, metadata)


if __name__ == "__main__":
    main()
