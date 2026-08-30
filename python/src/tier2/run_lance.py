"""
tier2/run_lance.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from lib.corpus_config import (
    CONCEPT_SETS,
    CORPUS_TIER2_DB_PATH,
    EVENTSTORE_T1_PATH,
    LANCE_INDEXES_DIR,
)
from lib.corpus_logging import logger
from retrieval.lance_observation_index_store import (
    LanceObservationIndexStore,
)
from retrieval.models import SearchSpace
from tier1.observation_store_api import (
    SCALES,
    open_observation_lookup,
)
from tier2.analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_concept_batches,
    resolve_concept_positions,
)
from tier2.sqlite import write_tier2_sqlite


def _resolve_search_scope(
    search_space: SearchSpace,
    lookup,
    scales=SCALES,
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    int | None,
    int | None,
]:
    available_years = {
        int(year)
        for year in lookup.available_years
    }

    candidate_years = tuple(
        search_space.resolve_years(available_years)
    )

    scales = tuple(
        search_space.resolve_scales(set(scales))
    )

    if not scales:
        raise ValueError(
            "SearchSpace resolves to no available scales"
        )

    if not candidate_years:
        logger.warning(
            "[tier2] SearchSpace resolves to no searchable years"
        )

    year_start = (
        min(candidate_years)
        if candidate_years
        else None
    )

    year_end = (
        max(candidate_years)
        if candidate_years
        else None
    )

    return (
        candidate_years,
        scales,
        year_start,
        year_end,
    )


def run_lance_tier2(
    *,
    concept_name: str,
    concept: dict,
    search_space: SearchSpace | None = None,
    store_path: str | Path = EVENTSTORE_T1_PATH,
    sqlite_path: str | Path = CORPUS_TIER2_DB_PATH,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
    clear: bool = False,
    lookup=None,
    indexes_by_year=None,
    candidate_years: tuple[int, ...] | None = None,
    scales: tuple[str, ...] | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
) -> Path:
    """
    Run Tier 2 semantic neighbourhood analysis using LanceDB.

    Tier 1 remains the source of truth for observation identity and
    provenance. Lance supplies approximate nearest-neighbour geometry.

    Lance tables cover the whole corpus physically. Temporal restriction is
    represented by one logical LanceObservationIndex per candidate year.

    Each seed event is therefore searched only against observations from
    the seed event's publication year.

    The observation lookup and temporal Lance indexes are shared between
    concepts by the CLI.

    Failure modes:
        Missing temporal indexes are rejected before search because a
        seed must never silently fall back to a broader temporal scope.

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

    if lookup is None:
        lookup = open_observation_lookup(
            store_path
        )

    if (
        candidate_years is None
        or scales is None
        or year_start is None
        or year_end is None
    ):
        (
            candidate_years,
            scales,
            year_start,
            year_end,
        ) = _resolve_search_scope(
            search_space,
            lookup,
            scales or SCALES,
        )

    if indexes_by_year is None:
        raise ValueError(
            "indexes_by_year must be supplied"
        )

    missing_years = set(candidate_years) - set(
        indexes_by_year
    )

    if missing_years:
        raise RuntimeError(
            "Missing temporal indexes for years: "
            f"{sorted(missing_years)}"
        )

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

    seed_ids = [
        event_id
        for year in candidate_years
        for event_id in resolved["by_year"].get(
            year,
            (),
        )
    ]

    logger.info(
        "[tier2] query workset: %d seed events, search years=%s-%s",
        len(seed_ids),
        year_start,
        year_end,
    )

    output_events = []

    search_started = time.perf_counter()
    batch_count = 0

    for batch in iter_concept_batches(
        lookup=lookup,
        indexes_by_year=indexes_by_year,
        seed_event_ids=seed_ids,
        scales=scales,
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
        time.perf_counter()
        - search_started
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
        time.perf_counter()
        - write_started
    )

    total_time = (
        time.perf_counter()
        - started
    )

    logger.info(
        "[tier2] concept=%s timing: search=%.3fs sqlite=%.3fs total=%.3fs",
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
        "--clear",
        action="store_true",
        help="Clear the Tier 2 SQLite database before processing.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="K nearest neighbours.",
    )

    parser.add_argument(
        "--oversample",
        type=int,
        default=OVERSAMPLE,
        help="K nearest neighbours oversample.",
    )

    parser.add_argument(
        "--concept",
        help="Run only this concept. Default: all CONCEPT_SETS entries.",
    )

    parser.add_argument(
        "--scale",
        choices=SCALES,
        action="append",
        help=(
            "Scale to build. May be supplied multiple times. "
            "Defaults to all scales."
        ),
    )

    parser.add_argument(
        "--sqlite",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
        help=(
            f"Tier 2 SQLite database "
            f"(default: {CORPUS_TIER2_DB_PATH})."
        ),
    )

    parser.add_argument(
        "--lance",
        type=Path,
        default=LANCE_INDEXES_DIR,
        help=(
            f"Lance database root "
            f"(default: {LANCE_INDEXES_DIR})."
        ),
    )

    parser.add_argument(
        "--store",
        type=Path,
        default=EVENTSTORE_T1_PATH,
        help=(
            f"Tier 1 observation store "
            f"(default: {EVENTSTORE_T1_PATH})."
        ),
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
        args_concept_norm = args.concept.upper()

        if args_concept_norm not in CONCEPT_SETS:
            available = ", ".join(
                sorted(CONCEPT_SETS)
            )

            raise SystemExit(
                f"Unknown concept {args_concept_norm!r}.\n"
                f"Available concepts: {available}"
            )

        concept_names = [args_concept_norm]

    else:
        concept_names = list(
            CONCEPT_SETS
        )

    scales = (
        tuple(args.scale)
        if args.scale
        else SCALES
    )

    search_space = SearchSpace(
        years=(
            (
                args.from_year,
                args.to_year,
            )
            if (
                args.from_year is not None
                or args.to_year is not None
            )
            else None
        ),
        scale=None,
    )

    logger.info( "[tier2] processing %d concept(s)", len(concept_names), )
    logger.info( "[tier2] SQLite output: %s", args.sqlite, )

    lookup = open_observation_lookup( args.store )

    (
        candidate_years,
        scales,
        year_start,
        year_end,
    ) = _resolve_search_scope(
        search_space,
        lookup,
        scales,
    )

    logger.info( "[tier2] SearchSpace years=%s scales=%s", candidate_years, scales, )

    db_started = time.perf_counter()

    index_store = LanceObservationIndexStore(
        args.lance,
        available_years=lookup.available_years,
        available_scales=scales,
    )

    logger.info( "[tier2] opened Lance observation index store in %.3fs", time.perf_counter() - db_started, )

    indexes_by_year = {
        year: index_store.get(
            SearchSpace(
                years=(year, year),
                scale=scales,
            )
        )
        for year in candidate_years
    }

    logger.info( "[tier2] prepared temporal indexes for %d year(s)", len(indexes_by_year), )

    for index, concept_name in enumerate(
        concept_names,
        start=1,
    ):
        logger.info( "[tier2] ===== concept %d/%d: %s =====", index, len(concept_names), concept_name, )

        # --clear is deliberately consumed only by the first concept.
        # Otherwise every concept would erase the results of its predecessor.
        clear = ( args.clear and index == 1 )

        run_lance_tier2(
            top_n=args.k,
            oversample=args.oversample,
            concept_name=concept_name,
            concept=CONCEPT_SETS[concept_name],
            search_space=search_space,
            store_path=args.store,
            sqlite_path=args.sqlite,
            clear=clear,
            lookup=lookup,
            indexes_by_year=indexes_by_year,
            candidate_years=candidate_years,
            scales=scales,
            year_start=year_start,
            year_end=year_end,
        )

    logger.info( "[tier2] completed %d concept(s)", len(concept_names) )


if __name__ == "__main__":
    main()
