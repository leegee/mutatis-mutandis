from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lib.corpus_config import (
    CONCEPT_SETS,
    EVENTSTORE_T1_PATH,
)
from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from retrieval.models import SearchSpace
from tier1.observation_store_api import open_observation_lookup
from tier2.diskann import run_diskann_tier2

DEFAULT_FORMS = ("hair",)
DEFAULT_TOP_N = 20
DEFAULT_OUTPUT = Path("out/test_phrase_search2.json")


def _parse_years(value: str) -> tuple[int, ...]:
    years: set[int] = set()

    for part in value.split(","):
        part = part.strip()

        if not part:
            continue

        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)

            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid year range: {part}"
                )

            years.update(range(start, end + 1))
        else:
            years.add(int(part))

    if not years:
        raise argparse.ArgumentTypeError(
            "At least one year is required"
        )

    return tuple(sorted(years))


def _parse_scales(values: list[str]) -> tuple[str, ...] | None:
    scales = {
        scale.strip()
        for value in values
        for scale in value.split(",")
        if scale.strip()
    }

    if not scales:
        return None

    invalid = scales - {"local", "medium", "broad"}

    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown scale(s): {', '.join(sorted(invalid))}; "
            "expected local, medium, broad"
        )

    return tuple(
        scale
        for scale in ("local", "medium", "broad")
        if scale in scales
    )


def _parse_forms(values: list[str]) -> tuple[str, ...]:
    forms = {
        form.strip()
        for value in values
        for form in value.split(",")
        if form.strip()
    }

    if not forms:
        raise argparse.ArgumentTypeError(
            "At least one form is required"
        )

    return tuple(sorted(forms))


def _get_document_titles(doc_ids):
    if not doc_ids:
        return {}

    with get_connection(
        application_name="test_phrase_search2"
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT doc_id, LEFT(title, 20) AS title
                FROM documents
                WHERE doc_id = ANY(%s)
                """,
                (list(doc_ids),),
            )
            return dict(cur.fetchall())


def _print_neighbour_table(events, limit: int = 100) -> None:
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


def _resolve_concept(
    name: str,
    extra_forms: list[str],
    extra_false_positives: list[str],
) -> tuple[dict, list[str]]:
    key = name.upper()

    try:
        rule = CONCEPT_SETS[key]
    except KeyError:
        available = ", ".join(sorted(CONCEPT_SETS))

        raise SystemExit(
            f"Unknown concept set: {name!r}\n"
            f"Available concept sets: {available}"
        )

    forms = set(rule["forms"])

    for value in extra_forms:
        forms.update(
            form.strip()
            for form in value.split(",")
            if form.strip()
        )

    false_positives = set(
        rule["false_positives"]
    )

    for value in extra_false_positives:
        false_positives.update(
            form.strip()
            for form in value.split(",")
            if form.strip()
        )

    if not forms:
        raise SystemExit(
            f"Concept set {key!r} contains no forms"
        )

    return (
        {"forms": sorted(forms)},
        sorted(false_positives),
    )


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Tier 2 DiskANN semantic neighbourhood search "
            "against the Tier 1 observation store."
        )
    )

    parser.add_argument(
        "concept",
        metavar="CONCEPT",
        help=( "Concept-set key from CONCEPT_SETS, for example LAW or PREROGATIVE." ),
    )

    parser.add_argument(
        "--forms",
        action="append",
        default=[],
        metavar="FORM[,FORM...]",
        help=( "Additional/override concept form(s). May be supplied multiple times or comma-separated." ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Number of neighbours to retain (default: {DEFAULT_TOP_N}).",
    )

    parser.add_argument(
        "--years",
        type=_parse_years,
        default=None,
        metavar="YEAR[,YEAR...]",
        help=( "Restrict the search to years. Supports ranges, e.g. 1580-1600, or combinations such as 1580-1590,1600,1605." ),
    )

    parser.add_argument(
        "--scale",
        action="append",
        default=[],
        metavar="SCALE",
        help=( "Scale(s) to search: local, medium, broad. May be repeated or comma-separated. Default: all scales." ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Override the Tier 2 concept batch size.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=None,
        metavar="N",
        help="Override reciprocal-rank-fusion k.",
    )

    parser.add_argument(
        "--oversample",
        type=int,
        default=None,
        metavar="N",
        help="Override DiskANN candidate oversampling.",
    )

    parser.add_argument(
        "--false-positive",
        action="append",
        default=[],
        metavar="FORM",
        help=( "Additional form to exclude as a false positive. May be supplied multiple times." ),
    )

    parser.add_argument(
        "--table-limit",
        type=int,
        default=10000,
        metavar="N",
        help="Maximum neighbour rows printed to the log (default: 100).",
    )

    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Do not print the neighbour table.",
    )

    return parser


def main() -> None:
    args = arg_parser().parse_args()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be greater than zero")

    if args.table_limit < 0:
        raise SystemExit("--table-limit must not be negative")

    concept, false_positives = _resolve_concept(
        args.concept,
        args.forms,
        args.false_positive,
    )

    scales = _parse_scales(args.scale)

    search_space = SearchSpace(
        years=args.years,
        scale=scales,
    )

    started = time.perf_counter()

    logger.debug("[test] Starting test_phrase_search2")
    logger.debug(
        "[test] concept=%s forms=%s false_positives=%s top_n=%d",
        args.concept.upper(),
        concept["forms"],
        false_positives,
        args.top_n,
    )
    logger.debug(
        "[test] SearchSpace: years=%s scales=%s",
        search_space.years,
        search_space.scale,
    )
    logger.debug("[test] Running Tier 2 semantic search")

    kwargs = {
        "concept_name": args.concept.upper(),
        "concept": concept,
        "output_path": args.output,
        "search_space": search_space,
        "top_n": args.top_n,
        "false_positives": false_positives,
    }

    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size

    if args.rrf_k is not None:
        kwargs["rrf_k"] = args.rrf_k

    if args.oversample is not None:
        kwargs["oversample"] = args.oversample

    t = time.perf_counter()

    output_path = run_diskann_tier2(**kwargs)

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

    if not args.no_table:
        _print_neighbour_table(
            events,
            limit=args.table_limit,
        )

    logger.debug( "[test] TOTAL test_phrase_search2 runtime: %.3fs", time.perf_counter() - started, )


if __name__ == "__main__":
    main()
