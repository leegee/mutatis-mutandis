from __future__ import annotations

import json
import time
from pathlib import Path

from lib.corpus_config import EVENTSTORE_T1_PATH
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.models import SearchSpace
from tier2.diskann import run_diskann_tier2
from tier1.observation_store_api import  open_observation_lookup

CONCEPT = {
    "forms": ["hair"],
}
TOP_N = 20
OUTPUT_PATH = Path("out/test_phrase_search2.json")


def _get_document_titles(doc_ids):
    if not doc_ids:
        return {}

    with get_connection(
        application_name="test_phrase_search2"
    ) as conn:
        with conn.cursor() as cur:
            cur.execute( "SELECT doc_id, LEFT(title, 20) AS title FROM documents WHERE doc_id = ANY(%s)",
                (list(doc_ids),),
            )
            return dict(cur.fetchall())


def _print_neighbour_table(events, limit=100):
    logger.debug(
        "\n[test]"
        "seed_event_id\tseed_idx\tneighbour_event_id\tdoc_id\ttitle\t"
        "token\ttoken_idx\tscore\tlocal\tmedium\tbroad\t"
        "local_window\tmedium_window\tbroad_window"
    )

    neighbour_metadata = {
        neighbour["event_id"]: neighbour
        for event in events
        for neighbour in event["neighbours"]
    }

    doc_ids = {
        neighbour["doc_id"]
        for neighbour in neighbour_metadata.values()
    }

    titles = _get_document_titles(doc_ids)

    rows = []

    for event in events:
        seed_event_id = event["event_id"]
        seed_idx = event["token_idx"]

        for neighbour in event["neighbours"]:
            doc_id = neighbour["doc_id"]

            rows.append(
                (
                    seed_event_id,
                    seed_idx,
                    neighbour["event_id"],
                    doc_id,
                    titles.get(doc_id),
                    neighbour["token"],
                    neighbour["token_idx"],
                    neighbour["score"],
                    neighbour["score_local"],
                    neighbour["score_medium"],
                    neighbour["score_broad"],
                    neighbour["local_window_id"],
                    neighbour["medium_window_id"],
                    neighbour["broad_window_id"],
                )
            )

    for row in rows[:limit]:
        logger.debug(
            "\t".join(
                "-" if value is None else str(value)
                for value in row
            )
        )


def main() -> None:
    started = time.perf_counter()

    logger.debug("[test] Starting test_phrase_search2")

    search_space = SearchSpace(
        years=None,
        scale=None,
    )

    logger.debug(
        "[test] SearchSpace: years=%s scales=%s",
        search_space.years,
        search_space.scale,
    )

    logger.debug("[test] Running all-years semantic search")

    t = time.perf_counter()

    output_path = run_diskann_tier2(
        concept_name="hair",
        concept=CONCEPT,
        output_path=OUTPUT_PATH,
        search_space=search_space,
        top_n=TOP_N,
    )

    logger.debug(
        "[test] run_diskann_tier2 completed in %.3fs: %s",
        time.perf_counter() - t,
        output_path,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        result = json.load(handle)

    events = result["events"]

    logger.debug( "[test] Search result: %d seed events", len(events), )

    logger.debug( "[test] Resolved years: %s", result.get("resolved_years"), )

    logger.debug( "[test] Resolved scales: %s", result.get("resolved_scales"), )

    _print_neighbour_table(events)

    logger.debug( "[test] TOTAL test_phrase_search2 runtime: %.3fs", time.perf_counter() - started, )


if __name__ == "__main__":
    main()
