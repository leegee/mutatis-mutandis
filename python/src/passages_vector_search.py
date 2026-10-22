
from pathlib import Path
import argparse

import numpy as np
import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.macberth import encode_passage_query
from lib.corpus_logging import logger

from passage_search_pipeline import passages_search as ps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search historical passages using a MacBERTh query vector."
    )
    parser.add_argument(
        "--query",
        default=(
            "his head and hairs were white as white wool, "
            "and as snow, and his eyes were as a flame"
        ),
        help="Text to encode as the search query.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1000,
        help="Earliest publication year (default: 1000).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2000,
        help="Latest publication year (default: 2000).",
    )
    parser.add_argument(
        "--representation",
        default="mean4",
        help="Passage representation (default: mean4).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum number of results (default: 30).",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Candidate limit (default: 10).",
    )

    args = parser.parse_args()

    if args.start_year > args.end_year:
        parser.error("--start-year must be <= --end-year")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.candidate_limit < 1:
        parser.error("--candidate-limit must be >= 1")

    return args


def normalise_vector(vector, description="query vector"):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if not np.isfinite(vector).all():
        raise ValueError(f"{description} contains NaN or infinity")

    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{description} has zero magnitude")

    return vector / norm


def main():
    args = parse_args()
    lance_dir = Path(LANCE_INDEXES_DIR)

    logger.info("Lance directory:", lance_dir)
    logger.info("Query:", args.query)
    logger.info(f"Years: {args.start_year}-{args.end_year}")
    logger.info("Representation:", args.representation)
    logger.info("Limit:", args.limit)
    logger.info("Candidate limit:", args.candidate_limit)

    db = lancedb.connect(str(lance_dir))

    query_vector = normalise_vector(
        encode_passage_query(args.query, args.representation),
        "Text query",
    )

    results = ps.search_passages(
        db=db,
        query_vector=query_vector,
        representation=args.representation,
        start_year=args.start_year,
        end_year=args.end_year,
        limit=args.limit,
        include_vectors=False,
    )

    if not results:
        logger.info("No results found.")
        return

    with get_connection(application_name="revelation-vector-test") as conn:
        for rank, result in enumerate(results, start=1):
            metadata = ps.fetch_passage_text(conn, result)

            logger.info(
                f"{rank}. year={result['pub_year']} "
                f"distance={result['distance']:.4f}"
            )
            logger.info(f"   doc_id={result['doc_id']}")
            logger.info(f"   title={metadata['title'] or '(untitled)'}")
            logger.info(f"   author={metadata['author'] or '(unknown)'}")
            logger.info(f"   filepath={metadata['filepath']}")
            logger.info(
                f"   tokens={result['token_start']}:"
                f"{result['token_end']}"
            )
            logger.info(f"   passage_id={result['passage_id']}")
            logger.info(
                f"   {metadata['passage_text'] or '[Text unavailable]'}"
            )


if __name__ == "__main__":
    main()
