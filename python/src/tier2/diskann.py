from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from lib.corpus_config import DISKANN_INDEXES_DIR, EVENTSTORE_T1_PATH
from lib.corpus_logging import logger
from retrieval.lazy_year_disk_ann import LazyYearDiskANN
from retrieval.models import SearchSpace
from tier1.observation_store_api import SCALES, open_observation_lookup

from .analysis import (
    BATCH_SIZE,
    K,
    OVERSAMPLE,
    RRF_K,
    iter_year_concept_batches,
    resolve_concept_positions,
)

DISKANN_DIMENSIONS = 768


def run_diskann_tier2(
    *,
    concept_name: str,
    concept: dict,
    output_path: str | Path,
    search_space: SearchSpace | None = None,
    store_path: str | Path = EVENTSTORE_T1_PATH,
    indexes_root: str | Path = DISKANN_INDEXES_DIR,
    top_n: int = K,
    rrf_k: int = RRF_K,
    oversample: int = OVERSAMPLE,
    batch_size: int = BATCH_SIZE,
    false_positives: list[str] | None = None,
) -> Path:
    """
    Run Tier 2 semantic neighbourhood analysis within a SearchSpace.

    Tier 1 remains the source of truth for observation identity and
    provenance. DiskANN supplies only approximate nearest-neighbour
    geometry.

    Processing is year-major: at most one year's geometric resources are
    resident at a time, while each year's seed events are searched before
    those resources are evicted.
    """
    started = time.perf_counter()

    output_path = Path(output_path)
    lookup = open_observation_lookup(store_path)

    if search_space is None:
        search_space = SearchSpace(
            years=None,
            scale=None,
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

    available_years = {
        int(year)
        for year in lookup.available_years
    }

    available_index_years = set(
        LazyYearDiskANN.available_years(
            indexes_root,
        )
    )

    searchable_years = (
        available_years
        & available_index_years
    )

    candidate_years = search_space.resolve_years(
        searchable_years,
    )

    scales = search_space.resolve_scales(
        set(SCALES),
    )

    if not scales:
        raise ValueError(
            "SearchSpace resolves to no available scales"
        )

    if not candidate_years:
        logger.warning(
            "[tier2] SearchSpace resolves to no searchable years"
        )

    years_to_process = tuple(
        year
        for year in candidate_years
        if year in resolved["by_year"]
    )

    missing_index_years = sorted(
        (
            set(resolved["by_year"])
            & available_years
        )
        - available_index_years
    )

    for year in missing_index_years:
        logger.warning( "[tier2] no DiskANN indexes for year=%s; seed events will be skipped", year, )

    logger.info( "[tier2] SearchSpace years=%s scales=%s", candidate_years, scales, )

    logger.info(
        "[tier2] query workset: %d years, %d seed events",
        len(years_to_process),
        sum(
            len(
                resolved["by_year"].get(
                    year,
                    (),
                )
            )
            for year in years_to_process
        ),
    )

    year_indexes = LazyYearDiskANN(
        indexes_root,
        candidate_years,
        dimensions=DISKANN_DIMENSIONS,
        num_threads=0,
        search_complexity=100,
        beam_width=2,
        batch_num_threads=0,
        num_nodes_to_cache=0,
    )

    token_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()

    output_events = []

    total_index_open_time = 0.0
    total_search_time = 0.0

    try:
        for year in years_to_process:
            year_started = time.perf_counter()

            seed_ids = resolved["by_year"].get( year, (), )

            if not seed_ids:
                continue

            logger.info( "[tier2] processing year=%s: %d seed events", year, len(seed_ids), )

            indexes_started = time.perf_counter()

            indexes = year_indexes.get( year, )

            index_open_time = ( time.perf_counter() - indexes_started )

            total_index_open_time += index_open_time

            logger.info( "[tier2] indexes ready for year=%s in %.3fs", year, index_open_time, )

            try:
                search_started = time.perf_counter()

                year_event_count = 0
                batch_count = 0

                for batch in iter_year_concept_batches(
                    lookup=lookup,
                    indexes=indexes,
                    year=year,
                    seed_event_ids=seed_ids,
                    scales=scales,
                    top_n=top_n,
                    rrf_k=rrf_k,
                    oversample=oversample,
                    false_positives=false_positives,
                    batch_size=batch_size,
                    token_counts=token_counts,
                    doc_counts=doc_counts,
                ):
                    events = batch["events"]

                    output_events.extend( events, )

                    year_event_count += len(events)
                    batch_count += 1

                search_time = ( time.perf_counter() - search_started )
                total_search_time += search_time

                logger.info( "[tier2] searched year=%s: %d seed events, %d batches, %d output events in %.3fs",
                    year,
                    len(seed_ids),
                    batch_count,
                    year_event_count,
                    search_time,
                )

            finally:
                year_indexes.evict(
                    year,
                )

            logger.info(
                "[tier2] completed year=%s in %.3fs",
                year,
                time.perf_counter()
                - year_started,
            )

    finally:
        year_indexes.close()

        logger.info(
            "[tier2] year_indexes closed",
        )

    output = {
        "concept": concept_name,
        "search_space": {
            "years": search_space.years,
            "scale": search_space.scale,
        },
        "resolved_years": list(candidate_years),
        "resolved_scales": list(scales),
        "forms": sorted(
            resolved["forms"],
        ),
        "false_positives": sorted(
            resolved["false_positives"],
        ),
        "events": output_events,
        "token_counts": dict(
            token_counts,
        ),
        "doc_counts": dict(
            doc_counts,
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_started = time.perf_counter()

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
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
        "[tier2] wrote %d events to %s in %.3fs",
        len(output_events),
        output_path,
        write_time,
    )

    logger.info(
        "[tier2] timing summary: "
        "index_open=%.3fs search=%.3fs write=%.3fs total=%.3fs",
        total_index_open_time,
        total_search_time,
        write_time,
        total_time,
    )

    return output_path
