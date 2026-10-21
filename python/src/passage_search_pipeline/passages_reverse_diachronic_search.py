# passage_search_pipeline/passages_reverse_diachronic_search.py


"""
passages_reverse_diachronic_search.py -
  Iteratively retrieve historically earlier passages through vector transfer.

The initial text query is encoded once. At each subsequent period, vectors
from retrieved passages search the immediately earlier period. Passage results
are deduplicated for the next frontier, but every source-to-target retrieval
is retained as a separate edge.

This module does not persist runs or establish historical semantic equivalence.
Retrieval distances indicate embedding similarity, not historical relevance.

    python src/passage_search_pipeline/passages_reverse_diachronic_search.py \
    --query "albino" \
    --periods "2000-2049,1950-1999,1900-1949,1850-1899" \
    --representation mean4 \
    --limit 20 \
    --top-k-per-step 5


"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from uuid import uuid4

import numpy as np
import lancedb


from lib.macberth import encode_passage_query
from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_logging import logger
from lib.corpus_db import get_connection

from passage_search_pipeline.passages_search import (
    REPRESENTATIONS,
    search_passages,
    fetch_passage_text,
)

class RetrievalEdge(TypedDict):
    run_id: str
    query_id: str
    step: int
    source_type: Literal["query", "passage"]
    source_id: str
    target_passage_id: str
    target_doc_id: str
    target_key: str
    target_corpus: str
    token_start: int
    token_end: int
    target_year: int
    start_year: int
    end_year: int
    representation: str
    distance: float
    retrieval_method: Literal["initial_query", "representation_transfer"]


@dataclass
class SearchRun:
    run_id: str
    query_id: str
    query: str
    representation: str
    periods: list[tuple[int, int]]
    limit: int
    top_k_per_step: int
    passages: dict[str, dict] = field(default_factory=dict)
    edges: list[RetrievalEdge] = field(default_factory=list)
    step_summaries: list[dict] = field(default_factory=list)


@dataclass
class FrontierItem:
    passage_key: str
    vector: np.ndarray


def passage_key(result: dict) -> str:
    """Identify a passage independently of the table that returned it."""
    return "|".join(
        str(result[field])
        for field in (
            "corpus",
            "doc_id",
            "token_start",
            "token_end",
        )
    )


def validate_periods(periods: list[tuple[int, int]]) -> None:
    if not periods:
        raise ValueError("At least one period is required")

    for start, end in periods:
        if start > end:
            raise ValueError(f"Invalid period: {start}-{end}")

    for current, earlier in zip(periods, periods[1:]):
        if earlier[0] > current[0] or earlier[1] >= current[0]:
            raise ValueError(
                "Periods must be ordered backwards and must not overlap"
            )


def reverse_diachronic_search(
    query: str,
    periods: list[tuple[int, int]],
    representation: str = "mean4",
    limit: int = 20,
    top_k_per_step: int = 5,
) -> SearchRun:
    """Search successive earlier periods using retrieved passage vectors.

    ``limit`` is the maximum number of results requested per source per
    period. ``top_k_per_step`` caps the number of results from each source
    that continue into the next period.

    A passage reached through multiple sources has one entry in ``passages``
    but retains every incoming retrieval in ``edges``.
    """
    if not query.strip():
        raise ValueError("Query must not be empty")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    if limit < 1:
        raise ValueError("limit must be positive")
    if top_k_per_step < 1:
        raise ValueError("top_k_per_step must be positive")

    validate_periods(periods)

    run_id = str(uuid4())
    query_id = str(uuid4())

    run = SearchRun(
        run_id=run_id,
        query_id=query_id,
        query=query,
        representation=representation,
        periods=list(periods),
        limit=limit,
        top_k_per_step=top_k_per_step,
    )

    query_vector = np.asarray(
        encode_passage_query(query, representation),
        dtype=np.float32,
    ).reshape(-1)

    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    frontier = [
        FrontierItem(
            passage_key=f"query:{query_id}",
            vector=query_vector,
        )
    ]
    frontier_sources = {
        f"query:{query_id}": {
            "source_type": "query",
            "source_id": query_id,
        }
    }

    for step, (start_year, end_year) in enumerate(periods):
        next_frontier: dict[str, FrontierItem] = {}
        next_sources: dict[str, dict] = {}
        edge_count = 0

        for item in frontier:
            source = frontier_sources[item.passage_key]

            results = search_passages(
                db,
                item.vector,
                representation,
                start_year,
                end_year,
                limit,
                include_vectors=True,
            )

            for result in results[:top_k_per_step]:
                key = passage_key(result)

                run.passages.setdefault(
                    key,
                    {
                        field: value
                        for field, value in result.items()
                        if field != "vector"
                    },
                )

                run.edges.append({
                    "run_id": run_id,
                    "query_id": query_id,
                    "step": step,
                    "source_type": source["source_type"],
                    "source_id": source["source_id"],
                    "target_passage_id": str(result["passage_id"]),
                    "target_doc_id": str(result["doc_id"]),
                    "target_year": int(result["pub_year"]),
                    "start_year": start_year,
                    "end_year": end_year,
                    "representation": representation,
                    "target_key": key,
                    "target_corpus": result["corpus"],
                    "token_start": result["token_start"],
                    "token_end": result["token_end"],
                    "distance": float(result["distance"]),
                    "retrieval_method": (
                        "initial_query"
                        if source["source_type"] == "query"
                        else "representation_transfer"
                    ),
                })
                edge_count += 1

                if key not in next_frontier:
                    vector = np.asarray(
                        result["vector"],
                        dtype=np.float32,
                    ).reshape(-1)

                    next_frontier[key] = FrontierItem(
                        passage_key=key,
                        vector=vector,
                    )
                    next_sources[key] = {
                        "source_type": "passage",
                        "source_id": str(result["passage_id"]),
                    }

        run.step_summaries.append({
            "step": step,
            "start_year": start_year,
            "end_year": end_year,
            "source_count": len(frontier),
            "edge_count": edge_count,
            "unique_passage_count": len(next_frontier),
        })

        frontier = list(next_frontier.values())
        frontier_sources = next_sources

        if not frontier:
            break

    return run


def main() -> None:
    parser = argparse.ArgumentParser( description="Iteratively search earlier periods using passage vectors." )
    parser.add_argument("--query", required=True)
    parser.add_argument( "--periods", required=True, help="Comma-separated inclusive ranges, newest first; e.g. 2000-2049,1950-1999,1900-1949", )
    parser.add_argument( "--representation", choices=REPRESENTATIONS, default="mean4", )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--top-k-per-step", type=int, default=5)
    args = parser.parse_args()

    try:
        periods = [
            tuple(map(int, part.split("-", maxsplit=1)))
            for part in args.periods.split(",")
        ]
        run = reverse_diachronic_search(
            query=args.query,
            periods=periods,
            representation=args.representation,
            limit=args.limit,
            top_k_per_step=args.top_k_per_step,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    logger.info(f"run_id={run.run_id} query_id={run.query_id}")
    logger.info(f"query={run.query!r}")
    logger.info(f"passages={len(run.passages)} edges={len(run.edges)}")

    for summary in run.step_summaries:
        logger.info(summary)

        with get_connection(application_name="reverse-diachronic-search") as conn:
            for summary in run.step_summaries:
                logger.info(summary)
                step = summary["step"]

                for edge in (e for e in run.edges if e["step"] == step):
                    passage = run.passages[edge["target_key"]]
                    text = fetch_passage_text(conn, passage)

                    logger.info(
                        f"  target={edge['target_corpus']}:{edge['target_doc_id']} "
                        f"tokens={edge['token_start']}:{edge['token_end']} "
                        f"passage_id={edge['target_passage_id']} "
                        f"source={edge['source_type']}:{edge['source_id']} "
                        f"distance={edge['distance']:.4f} "
                        f"year={edge['target_year']} "
                        f"title={text['title']!r} "
                        f"author={text['author']!r}"
                    )
                    logger.info(f"    {text['passage_text']}")

if __name__ == "__main__":
    main()
