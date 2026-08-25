"""
tier2/run_lance.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lancedb

from lib.corpus_config import EVENTSTORE_T1_PATH, CONCEPT_SETS, LANCE_INDEXES_DIR, CORPUS_TIER2_DB_URL
from lib.corpus_logging import logger
from retrieval.models import SearchSpace
from tier1.observation_store_api import SCALES, open_observation_lookup

from tier2.analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_concept_batches,
    resolve_concept_positions,
)
from tier2.sqlite import write_tier2_sqlite



def run_lance_tier2(
    *,
    concept_name: str,
    concept: dict,
    search_space: SearchSpace | None = None,
    store_path: str | Path = EVENTSTORE_T1_PATH,
    lance_root: str | Path = LANCE_INDEXES_DIR,
    sqlite_path: str | Path = CORPUS_TIER2_DB_URL,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
    clear: bool = False,
) -> Path:
    """
    Run Tier 2 semantic neighbourhood analysis using LanceDB.

    Tier 1 remains the source of truth for observation identity and
    provenance. Lance supplies approximate nearest-neighbour geometry.

    The Lance database contains one table per embedding scale covering the
    corpus. Temporal restriction is performed at query time.

    Results are written to the established Tier 2 SQLite schema. No JSON
    intermediate is produced.

    Failure mode:
        The complete result set is accumulated in memory before the SQLite
        transaction. This is intentionally retained for compatibility with
        the existing analysis layer; streaming persistence can be introduced
        after downstream consumers have been checked.
    """
    started = time.perf_counter()

    if search_space is None:
        search_space = SearchSpace(
            years=None,
            scale=None,
        )

    lookup = open_observation_lookup(store_path)

    logger.info(
        "[tier2] resolving concept=%s",
        concept_name,
    )

    resolve_started = time.perf_counter()

    resolved = resolve_concept_positions(
        concept_name=concept_name,
        concept=concept,
        lookup=lookup,
        false_positives=false_positives,
    )

    logger.info(
        "[tier2] resolved concept=%s in %.3fs",
        concept_name,
        time.perf_counter() - resolve_started,
    )

    available_years = {
        int(year)
        for year in lookup.available_years
    }

    candidate_years = tuple(
        search_space.resolve_years(
            available_years,
        )
    )

    scales = tuple(
        search_space.resolve_scales(
            set(SCALES),
        )
    )

    if not scales:
        raise ValueError(
            "SearchSpace resolves to no available scales"
        )

    if not candidate_years:
        logger.warning(
            "[tier2] SearchSpace resolves to no searchable years"
        )

    year_start = min(candidate_years) if candidate_years else None
    year_end = max(candidate_years) if candidate_years else None

    seed_ids = [
        event_id
        for year in candidate_years
        for event_id in resolved["by_year"].get(year, ())
    ]

    logger.info(
        "[tier2] SearchSpace years=%s scales=%s",
        candidate_years,
        scales,
    )

    logger.info(
        "[tier2] query workset: %d seed events, search years=%s-%s",
        len(seed_ids),
        year_start,
        year_end,
    )

    db_started = time.perf_counter()

    db = lancedb.connect(
        str(lance_root),
    )

    indexes = {}

    for scale in scales:
        try:
            indexes[scale] = db.open_table(scale)
        except Exception as exc:
            raise RuntimeError(
                f"Could not open Lance table for scale={scale}: "
                f"{lance_root}"
            ) from exc

    logger.info(
        "[tier2] opened %d Lance scale tables in %.3fs",
        len(indexes),
        time.perf_counter() - db_started,
    )

    output_events = []

    search_started = time.perf_counter()
    batch_count = 0

    for batch in iter_concept_batches(
        lookup=lookup,
        indexes=indexes,
        seed_event_ids=seed_ids,
        scales=scales,
        year_start=year_start,
        year_end=year_end,
        top_n=top_n,
        rrf_k=rrf_k,
        oversample=oversample,
        false_positives=false_positives,
        batch_size=batch_size,
    ):
        output_events.extend(
            batch["events"]
        )
        batch_count += 1

    search_time = (
        time.perf_counter() - search_started
    )

    logger.info(
        "[tier2] search complete: %d batches, %d seed events, %.3fs",
        batch_count,
        len(output_events),
        search_time,
    )

    write_started = time.perf_counter()

    sqlite_path = Path(sqlite_path)

    write_tier2_sqlite(
        db_path=sqlite_path,
        concept_name=concept_name,
        events=output_events,
        clear=clear,
    )

    write_time = (
        time.perf_counter() - write_started
    )

    total_time = (
        time.perf_counter() - started
    )

    logger.info(
        "[tier2] concept=%s timing: "
        "search=%.3fs sqlite=%.3fs total=%.3fs",
        concept_name,
        search_time,
        write_time,
        total_time,
    )

    return sqlite_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier 2 semantic neighbourhood analysis."
    )

    parser.add_argument(
        "--concept",
        help="Run only this concept. Default: all CONCEPT_SETS entries.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the Tier 2 SQLite database before processing.",
    )

    parser.add_argument(
        "--sqlite",
        type=Path,
        default=CORPUS_TIER2_DB_URL,
        help=f"Tier 2 SQLite database (default: {CORPUS_TIER2_DB_URL}).",
    )

    parser.add_argument(
        "--lance",
        type=Path,
        default=LANCE_INDEXES_DIR,
        help=f"Lance database root (default: {LANCE_INDEXES_DIR}).",
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=EVENTSTORE_T1_PATH,
        help=f"Tier 1 observation store (default: {EVENTSTORE_T1_PATH}).",
    )

    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Restrict retrieval to this publication year or later.",
    )

    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="Restrict retrieval to this publication year or earlier.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.concept is not None:
        if args.concept not in CONCEPT_SETS:
            available = ", ".join(
                sorted(CONCEPT_SETS)
            )
            raise SystemExit(
                f"Unknown concept {args.concept!r}.\n"
                f"Available concepts: {available}"
            )

        concept_names = [args.concept]

    else:
        concept_names = list(CONCEPT_SETS)

    if args.from_year is not None or args.to_year is not None:
        search_space = SearchSpace(
            years=(
                args.from_year,
                args.to_year,
            ),
            scale=None,
        )
    else:
        search_space = SearchSpace(
            years=None,
            scale=None,
        )

    logger.info(
        "[tier2] processing %d concept(s)",
        len(concept_names),
    )

    logger.info(
        "[tier2] SQLite output: %s",
        args.sqlite,
    )

    for index, concept_name in enumerate(
        concept_names,
        start=1,
    ):
        logger.info(
            "[tier2] ===== concept %d/%d: %s =====",
            index,
            len(concept_names),
            concept_name,
        )

        # --clear is deliberately consumed only by the first concept.
        # Otherwise every concept would erase the results of its predecessor.
        clear = args.clear and index == 1

        run_lance_tier2(
            concept_name=concept_name,
            concept=CONCEPT_SETS[concept_name],
            search_space=search_space,
            store_path=args.store,
            lance_root=args.lance,
            sqlite_path=args.sqlite,
            clear=clear,
        )

    logger.info(
        "[tier2] completed %d concept(s)",
        len(concept_names),
    )


if __name__ == "__main__":
    main()
