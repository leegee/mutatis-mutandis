from __future__ import annotations

import time
from collections import Counter

from lib.corpus_config import EVENTSTORE_T1_PATH, DISKANN_INDEXES_DIR
from lib.corpus_logging import logger
from tier1.observation_store_api import open_observation_lookup
from retrieval.lazy_year_disk_ann import LazyYearDiskANN
from tier2.analysis import iter_year_concept_batches


CONCEPT = {
    "forms": ["hair"],
}

YEAR = 1676
TOP_N = 20


def main() -> None:
    started = time.perf_counter()

    logger.debug("[test] Starting test_phrase_search2")

    logger.debug(
        "[test] Opening observation lookup: %s",
        EVENTSTORE_T1_PATH,
    )
    t = time.perf_counter()
    lookup = open_observation_lookup(EVENTSTORE_T1_PATH)
    logger.debug(
        "[test] Observation lookup opened in %.3fs; %d observations; years=%d..%d",
        time.perf_counter() - t,
        len(lookup),
        int(lookup.available_years.min())
        if len(lookup.available_years)
        else -1,
        int(lookup.available_years.max())
        if len(lookup.available_years)
        else -1,
    )

    logger.debug(
        "[test] Available years: %s",
        lookup.available_years.tolist(),
    )

    logger.debug("[test] Finding seed events for forms=%s", CONCEPT["forms"])
    t = time.perf_counter()

    event_ids = lookup.find_matching_event_ids(
        CONCEPT["forms"],
    )

    logger.debug(
        "[test] Found %d seed events in %.3fs",
        len(event_ids),
        time.perf_counter() - t,
    )

    year_counts = Counter(
        int(lookup.pub_year[lookup.get_pos(eid)])
        for eid in event_ids
    )

    logger.debug(
        "[test] Seed-event year distribution: %s",
        dict(sorted(year_counts.items())),
    )

    year_events = [
        eid
        for eid in event_ids
        if int(lookup.pub_year[lookup.get_pos(eid)]) == YEAR
    ]

    logger.debug(
        "[test] Seed events in year %d: %d",
        YEAR,
        len(year_events),
    )

    if year_events:
        first_eid = year_events[0]

        logger.debug(
            "[test] First %d event metadata: %s",
            first_eid,
            lookup.get_event_metadata(first_eid),
        )

        t = time.perf_counter()

        embedding = lookup.get_ensemble_embedding(
            lookup.get_pos(first_eid),
        )

        logger.debug(
            "[test] First event ensemble embedding loaded in %.3fs: "
            "shape=%s dtype=%s norm=%.6f",
            time.perf_counter() - t,
            embedding.shape,
            embedding.dtype,
            float((embedding ** 2).sum() ** 0.5),
        )

    logger.debug(
        "[test] Opening LazyYearDiskANN: %s",
        DISKANN_INDEXES_DIR,
    )
    t = time.perf_counter()

    indexes = LazyYearDiskANN(
        DISKANN_INDEXES_DIR,
        lookup.available_years,
        dimensions=768,
        num_threads=0,
        search_complexity=100,
        beam_width=2,
        batch_num_threads=0,
        num_nodes_to_cache=0,
    )

    logger.debug(
        "[test] LazyYearDiskANN constructed in %.3fs",
        time.perf_counter() - t,
    )

    logger.debug(
        "[test] Starting iter_year_concept_batches: "
        "concept=%r year=%d top_n=%d",
        "hair",
        YEAR,
        TOP_N,
    )

    iteration_started = time.perf_counter()
    batch_count = 0
    event_count = 0

    for batch in iter_year_concept_batches(
        concept_name="hair",
        concept=CONCEPT,
        lookup=lookup,
        indexes=indexes,
        year=YEAR,
        top_n=TOP_N,
    ):
        batch_count += 1

        events = batch["events"]
        event_count += len(events)

        logger.debug(
            "[test] Received batch %d after %.3fs: %d events; keys=%s",
            batch_count,
            time.perf_counter() - iteration_started,
            len(events),
            sorted(batch.keys()),
        )

        for event in events:
            logger.debug(
                "[test] EVENT: %s",
                event,
            )

    iteration_elapsed = time.perf_counter() - iteration_started

    logger.debug(
        "[test] iter_year_concept_batches completed in %.3fs: "
        "batches=%d events=%d",
        iteration_elapsed,
        batch_count,
        event_count,
    )

    logger.debug(
        "[test] TOTAL test_phrase_search2 runtime: %.3fs",
        time.perf_counter() - started,
    )


if __name__ == "__main__":
    main()
