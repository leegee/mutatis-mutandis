#!/usr/bin/env python
"""
passage_search_exp.py

Experimental multi-feature search over existing MacBERTh passage
embeddings in LanceDB, with manually seeded diachronic retrieval.

Multi-feature mode:
    1. Encode a broad query and each appearance feature independently.
    2. Retrieve candidates independently for every query.
    3. Union candidates by passage_id.
    4. Score each candidate against every query vector.
    5. Rank using:
           combined_score = 0.5 * broad_similarity
                          + 0.5 * mean(top two feature similarities)

Legacy single-query mode:
    --query TEXT

Recursive mode:
    --seed DOC_ID:TOKEN_START [--seed ...]
    --periods "1650-1699,1600-1649"

A seed is the token_start of an indexed passage, not an arbitrary token
inside a passage. Recursive mode performs one hop per invocation; review
the returned passages and explicitly select seeds for the next invocation.

Scores are exploratory ranking signals, not calibrated probabilities.

    python src/passage_search_pipeline/passage_search_exp.py \
    --start-year 1700 \
    --end-year 2020

    python src/passage_search_pipeline/passage_search_exp.py \
    --seed ACTUAL_DOC_ID:ACTUAL_TOKEN_START \
    --periods "1650-1699,1600-1649,1550-1599" \
    --representation mean4 \
    --recursive-limit 20

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lancedb
import numpy as np

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_BUCKET_SIZE,
    LANCE_MODEL_NAME,
)
from lib.corpus_db import get_connection
from lib.macberth import PASSAGE_REPRESENTATIONS, encode_passage_query
from tier1.vector_writer import year_bucket


REPRESENTATIONS = PASSAGE_REPRESENTATIONS

DEFAULT_FEATURES = {
    "white_hair": "naturally white hair",
    "pale_skin": "exceptionally white or milk-white skin",
    "white_eyebrows": "white eyebrows and white eyelashes",
    "unusual_eyes": "unusual eye colour or appearance",
}

VALID_LABELS = {"good", "promising", "bad"}


def passage_key(result: dict) -> str:
    """Canonical CLI identifier: document ID plus passage token_start."""
    return f"{result['doc_id']}:{int(result['token_start'])}"


def parse_seed(value: str) -> tuple[str, int]:
    """Parse DOC_ID:TOKEN_START."""
    try:
        doc_id, token_idx_text = value.rsplit(":", 1)
        token_idx = int(token_idx_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid seed {value!r}; expected DOC_ID:TOKEN_START"
        ) from exc

    if not doc_id or token_idx < 0:
        raise argparse.ArgumentTypeError(
            f"Invalid seed {value!r}; expected "
            "DOC_ID:non-negative-token-start"
        )

    return doc_id, token_idx


def parse_label(value: str) -> tuple[str, str]:
    """Parse DOC_ID:TOKEN_START=good|promising|bad."""
    try:
        seed_text, label = value.rsplit("=", 1)
        doc_id, token_idx = parse_seed(seed_text)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise argparse.ArgumentTypeError(
            "Expected DOC_ID:TOKEN_START=good|promising|bad"
        ) from exc

    if label not in VALID_LABELS:
        raise argparse.ArgumentTypeError(
            f"Unknown label {label!r}; use good, promising or bad"
        )

    return f"{doc_id}:{token_idx}", label


def parse_periods(value: str) -> list[tuple[int, int]]:
    """Parse comma-separated inclusive year ranges in search order."""
    periods: list[tuple[int, int]] = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        try:
            start_text, end_text = item.split("-", 1)
            start_year, end_year = int(start_text), int(end_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid period {item!r}; expected START-END"
            ) from exc

        if start_year > end_year:
            raise argparse.ArgumentTypeError(
                f"Period start exceeds end: {item!r}"
            )

        periods.append((start_year, end_year))

    if not periods:
        raise argparse.ArgumentTypeError(
            "Specify at least one period, e.g. 1650-1699,1600-1649."
        )

    return periods


def table_name(representation: str, year: int) -> str:
    start, end = year_bucket(year, LANCE_BUCKET_SIZE)
    return (
        f"passage_{representation}__{LANCE_MODEL_NAME}"
        f"__{start:04d}_{end:04d}"
    )


def candidate_tables(db, representation, start_year, end_year):
    """Return existing index tables overlapping the requested years."""
    listing = db.list_tables()

    if hasattr(listing, "tables"):
        available = set(listing.tables)
    else:
        available = set(listing)

    names = {
        table_name(representation, year)
        for year in range(start_year, end_year + 1)
    }

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
    """
    Retrieve and globally rank nearest passages for one query.

    `limit` applies per table before the hits are merged and globally
    truncated. Candidate recall can therefore depend on the number of
    year-bucket tables.
    """
    if limit < 1:
        raise ValueError("Search limit must be positive.")

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
            .limit(limit)
            .to_arrow()
        )

        for row in arrow.to_pylist():
            distance = row.get("_distance")

            if distance is None:
                raise RuntimeError(
                    f"Lance search returned no _distance "
                    f"for table {name!r}."
                )

            row["distance"] = float(distance)
            row["table"] = name
            results.append(row)

    results.sort(key=lambda row: row["distance"])
    return results[:limit]


def get_seed_passage(
    db,
    doc_id: str,
    token_idx: int,
    representation: str,
) -> dict:
    """
    Retrieve an exact indexed passage and its stored vector.

    token_idx must be the passage's token_start. Duplicate copies of
    the same passage ID across tables are collapsed; genuinely
    ambiguous identifiers are rejected.
    """
    prefix = f"passage_{representation}__{LANCE_MODEL_NAME}__"
    listing = db.list_tables()

    if hasattr(listing, "tables"):
        available = set(listing.tables)
    else:
        available = set(listing)

    matches = []

    escaped_doc_id = doc_id.replace("'", "''")
    predicate = (
        f"doc_id = '{escaped_doc_id}' "
        f"AND token_start = {int(token_idx)}"
    )

    for name in sorted(
        table_name for table_name in available
        if table_name.startswith(prefix)
    ):
        table = db.open_table(name)

        rows = (
            table.search()
            .where(predicate)
            .select([
                "passage_id",
                "corpus",
                "doc_id",
                "pub_year",
                "word_start",
                "word_end",
                "token_start",
                "token_end",
                "embedding_model",
                "vector",
            ])
            .limit(2)
            .to_list()
        )

        matches.extend(rows)

    if not matches:
        raise ValueError(
            f"No indexed passage found for {doc_id}:{token_idx} "
            f"using representation {representation!r}. Check that "
            "token_idx is the passage's token_start and that its "
            "vector is indexed."
        )

    by_id = {
        row.get("passage_id"): row
        for row in matches
    }

    if len(by_id) != 1:
        details = [
            (row.get("passage_id"), row.get("pub_year"))
            for row in matches
        ]

        raise ValueError(
            f"Ambiguous seed {doc_id}:{token_idx}; matches: {details}"
        )

    result = next(iter(by_id.values()))
    vector = result.get("vector")

    if vector is None:
        raise ValueError(
            f"Stored vector missing for {doc_id}:{token_idx}"
        )

    result["vector"] = normalise_vector(
        vector,
        f"seed passage {doc_id}:{token_idx}",
    )

    return result


def fetch_passage_text(conn, result):
    """Reconstruct passage text and retrieve document metadata."""
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
            JOIN tokens t
              ON t.corpus = d.corpus
             AND t.doc_id = d.doc_id
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


def recursive_seed_search(
    db,
    seeds: list[tuple[str, int]],
    representation: str,
    periods: list[tuple[int, int]],
    limit: int = 20,
) -> list[dict]:
    """
    Search requested periods using selected seeds' stored vectors.

    This is one hop per invocation, not an automatic traversal.
    """
    if limit < 1:
        raise ValueError("Recursive limit must be positive.")

    edges = []
    seen_edges = set()

    for doc_id, token_idx in seeds:
        seed_id = f"{doc_id}:{token_idx}"

        seed = get_seed_passage(
            db,
            doc_id,
            token_idx,
            representation,
        )

        query_vector = seed["vector"]

        for start_year, end_year in periods:
            matches = search_passages(
                db=db,
                query_vector=query_vector,
                representation=representation,
                start_year=start_year,
                end_year=end_year,
                limit=limit,
                include_vectors=False,
            )

            for match in matches:
                match_id = passage_key(match)
                edge_key = (
                    seed_id,
                    match_id,
                    start_year,
                    end_year,
                )

                if edge_key in seen_edges:
                    continue

                seen_edges.add(edge_key)

                edges.append({
                    "seed_id": seed_id,
                    "seed_passage_id": seed.get("passage_id"),
                    "match_id": match_id,
                    "match_passage_id": match.get("passage_id"),
                    "period_start": start_year,
                    "period_end": end_year,
                    "distance": match["distance"],
                    "result": match,
                })

    edges.sort(key=lambda edge: edge["distance"])
    return edges


def print_recursive_results(conn, edges: list[dict]) -> None:
    for rank, edge in enumerate(edges, start=1):
        result = edge["result"]
        metadata = fetch_passage_text(conn, result)

        print()
        print("=" * 90)
        print(
            f"{rank}. year={result.get('pub_year')} "
            f"distance={edge['distance']:.4f} "
            f"period={edge['period_start']}-{edge['period_end']}"
        )
        print(f"   parent_seed={edge['seed_id']}")
        print(f"   passage_id={edge['match_passage_id']}")
        print(
            f"   corpus={result['corpus']} "
            f"doc_id={result['doc_id']}"
        )
        print(f"   seed_id={edge['match_id']}")
        print(f"   title={metadata['title'] or '(untitled)'}")
        print(f"   author={metadata['author'] or '(unknown)'}")
        print(
            f"   tokens={result['token_start']}:{result['token_end']} "
            f"words={result['word_start']}:{result['word_end']}"
        )
        print(f"   filepath={metadata['filepath']}")
        print(
            f"   {metadata['passage_text'] or '[Text unavailable]'}"
        )
        print(f"   Next seed: --seed {edge['match_id']}")


def save_labels(
    path: Path,
    labels: list[tuple[str, str]],
) -> None:
    """Upsert labels into a CSV keyed by passage identifier."""
    existing: dict[str, str] = {}

    if path.exists():
        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as handle:
            for row in csv.DictReader(handle):
                key = row.get("passage_id_key")
                label = row.get("label")

                if key and label:
                    existing[key] = label

    for passage_id_key, label in labels:
        existing[passage_id_key] = label

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["passage_id_key", "label"],
        )
        writer.writeheader()

        for passage_id_key in sorted(existing):
            writer.writerow({
                "passage_id_key": passage_id_key,
                "label": existing[passage_id_key],
            })


def parse_features(args, parser):
    """Choose default or custom feature queries."""
    if args.feature:
        features = {}

        for item in args.feature:
            name, separator, text = item.partition("=")
            name = name.strip()
            text = text.strip()

            if not separator or not name or not text:
                parser.error(
                    f"Invalid --feature {item!r}; use NAME=TEXT."
                )

            if name == "broad":
                parser.error(
                    "'broad' is reserved; choose another feature name."
                )

            if name in features:
                parser.error(
                    f"Duplicate feature name: {name!r}"
                )

            features[name] = text

        return features

    return DEFAULT_FEATURES.copy()


def normalise_vector(vector, description):
    """Convert a vector to a finite, unit-length float32 array."""
    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"Non-finite values in vector for {description}."
        )

    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(
            f"Invalid or zero vector for {description}."
        )

    return vector / norm


def multi_feature_search(
    db,
    features: dict[str, str],
    broad_query: str,
    representation: str,
    start_year: int,
    end_year: int,
    limit: int = 20,
    candidate_limit: int = 100,
) -> list[dict]:
    """Retrieve candidates per query, union by passage_id, and rescore."""
    if not features:
        raise ValueError("At least one feature query is required.")

    if candidate_limit < 1:
        raise ValueError("Candidate limit must be positive.")

    queries = {"broad": broad_query, **features}

    query_vectors = {
        name: normalise_vector(
            encode_passage_query(query_text, representation),
            f"query {name!r}: {query_text!r}",
        )
        for name, query_text in queries.items()
    }

    candidates: dict[str, dict] = {}
    retrieved_by: dict[str, set[str]] = {}

    for name, query_vector in query_vectors.items():
        hits = search_passages(
            db=db,
            query_vector=query_vector,
            representation=representation,
            start_year=start_year,
            end_year=end_year,
            limit=candidate_limit,
            include_vectors=True,
        )

        for hit in hits:
            passage_id = hit["passage_id"]

            if passage_id not in candidates:
                candidates[passage_id] = hit
                retrieved_by[passage_id] = set()

            retrieved_by[passage_id].add(name)

    results = []

    for passage_id, hit in candidates.items():
        passage_vector = normalise_vector(
            hit["vector"],
            f"passage {passage_id!r}",
        )

        similarities = {
            name: float(
                np.clip(
                    np.dot(passage_vector, query_vector),
                    -1.0,
                    1.0,
                )
            )
            for name, query_vector in query_vectors.items()
        }

        strongest = sorted(
            (similarities[name] for name in features),
            reverse=True,
        )[:2]

        broad_similarity = similarities["broad"]
        feature_score = float(np.mean(strongest))

        combined_score = (
            0.5 * broad_similarity
            + 0.5 * feature_score
        )

        results.append({
            **hit,
            "similarities": similarities,
            "broad_similarity": broad_similarity,
            "feature_score": feature_score,
            "combined_score": combined_score,
            "retrieved_by": sorted(
                retrieved_by[passage_id]
            ),
        })

    results.sort(
        key=lambda row: row["combined_score"],
        reverse=True,
    )

    return results[:limit]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Experimental multi-feature MacBERTh passage search."
    )

    parser.add_argument(
        "--seed",
        action="append",
        type=parse_seed,
        default=[],
        metavar="DOC_ID:TOKEN_START",
        help=(
            "Seed recursive search with an indexed passage; repeatable."
        ),
    )

    parser.add_argument(
        "--label",
        action="append",
        type=parse_label,
        default=[],
        metavar="DOC_ID:TOKEN_START=LABEL",
        help="Save a label: good, promising or bad; repeatable.",
    )

    parser.add_argument(
        "--labels-file",
        type=Path,
        default=Path("out/passage_search_labels.csv"),
        help=(
            "CSV file used to persist labels "
            "(default: out/passage_search_labels.csv)."
        ),
    )

    parser.add_argument(
        "--periods",
        default="",
        help=(
            'Recursive target periods, e.g. "1650-1699,1600-1649".'
        ),
    )

    parser.add_argument(
        "--recursive-limit",
        type=int,
        default=20,
        help="Maximum matches returned per seed per period.",
    )

    parser.add_argument(
        "--query",
        help=(
            "Run legacy single-query search instead of multi-feature search."
        ),
    )

    parser.add_argument(
        "--broad-query",
        default="albino",
        help=(
            "Broad query for multi-feature mode (default: albino)."
        ),
    )

    parser.add_argument(
        "--feature",
        action="append",
        help=(
            "Custom feature NAME=TEXT; repeat to replace the default features."
        ),
    )

    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)

    parser.add_argument(
        "--representation",
        choices=REPRESENTATIONS,
        default="mean4",
    )

    parser.add_argument("--limit", type=int, default=20)

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=100,
        help=(
            "Per-table retrieval limit for each query before global "
            "truncation; the final candidate list is globally truncated "
            "to this value."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be positive")

    if args.candidate_limit < 1:
        parser.error("--candidate-limit must be positive")

    if args.recursive_limit < 1:
        parser.error("--recursive-limit must be positive")

    if args.query and args.feature:
        parser.error("Use either --query or --feature, not both.")

    if args.query and args.broad_query != "albino":
        parser.error(
            "--broad-query cannot be combined with legacy --query."
        )

    if args.seed:
        if not args.periods:
            parser.error("--seed requires --periods")

        try:
            periods = parse_periods(args.periods)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    else:
        if args.start_year is None or args.end_year is None:
            parser.error(
                "Initial search requires --start-year and --end-year."
            )

        if args.start_year > args.end_year:
            parser.error("--start-year must not exceed --end-year")

        if args.candidate_limit < args.limit:
            parser.error("--candidate-limit must be >= --limit")

    if args.label:
        save_labels(args.labels_file, args.label)
        print(f"Labels saved to {args.labels_file}")

    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    if args.seed:
        edges = recursive_seed_search(
            db=db,
            seeds=args.seed,
            representation=args.representation,
            periods=periods,
            limit=args.recursive_limit,
        )

        if not edges:
            print("No passages found in the requested periods.")
            return

        with get_connection(
            application_name="passage-search-exp"
        ) as conn:
            print_recursive_results(conn, edges)

        return

    if args.query:
        query_vector = normalise_vector(
            encode_passage_query(
                args.query,
                args.representation,
            ),
            f"query {args.query!r}",
        )

        results = search_passages(
            db=db,
            query_vector=query_vector,
            representation=args.representation,
            start_year=args.start_year,
            end_year=args.end_year,
            limit=args.limit,
        )

    else:
        features = parse_features(args, parser)

        print(f"Broad query: {args.broad_query}")
        print("Feature queries:")

        for name, text in features.items():
            print(f"  {name}: {text}")

        results = multi_feature_search(
            db=db,
            features=features,
            broad_query=args.broad_query,
            representation=args.representation,
            start_year=args.start_year,
            end_year=args.end_year,
            limit=args.limit,
            candidate_limit=args.candidate_limit,
        )

    if not results:
        print("No passages found for this date range.")
        return

    with get_connection(
        application_name="passage-search-exp"
    ) as conn:
        for rank, result in enumerate(results, start=1):
            metadata = fetch_passage_text(conn, result)

            print()

            if "combined_score" in result:
                print(
                    f"{rank}. year={result['pub_year']} "
                    f"combined={result['combined_score']:.4f} "
                    f"broad={result['broad_similarity']:.4f} "
                    f"features={result['feature_score']:.4f}"
                )

                for name, similarity in result["similarities"].items():
                    print(
                        f"   {name}: similarity={similarity:.4f} "
                        f"cosine_distance={1.0 - similarity:.4f}"
                    )

                print(
                    "   retrieved_by="
                    + ", ".join(result["retrieved_by"])
                )

            else:
                print(
                    f"{rank}. year={result['pub_year']} "
                    f"distance={result['distance']:.4f}"
                )

            print(f"   seed_id={passage_key(result)}")

            print(
                f"   corpus={result['corpus']} "
                f"doc_id={result['doc_id']} "
                f"passage_id={result['passage_id']}"
            )

            print(f"   title={metadata['title'] or '(untitled)'}")

            print(
                f"   words={result['word_start']}:{result['word_end']} "
                f"tokens={result['token_start']}:{result['token_end']}"
            )

            print(f"   author={metadata['author'] or '(unknown)'}")
            print(f"   filepath={metadata['filepath']}")
            print(
                f"   {metadata['passage_text'] or '[Text unavailable]'}"
            )

            print(f"   Next seed: --seed {passage_key(result)}")


if __name__ == "__main__":
    main()
