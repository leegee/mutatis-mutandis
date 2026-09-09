from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import lancedb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.corpus_config import LANCE_INDEXES_DIR
from lib.corpus_db import get_connection
from lib.macberth import MACBERTH_MODEL_NAME


K = 60
OVERSAMPLE = 2
NPROBES = 20
DIMENSIONS = 768

TABLE_PATTERN = re.compile(
    r"^(?P<scale>local|medium|broad)"
    r"__(?P<model>[^_]+(?:_[^_]+)*)"
    r"__(?P<year_start>\d{4})_(?P<year_end>\d{4})$"
)


def get_white_events_by_year() -> dict[int, list[int]]:
    """Return every current WHITE event grouped by authoritative publication year."""

    query = """
        SELECT
            e.pub_year,
            e.event_id
        FROM events e
        WHERE lower(e.token) = 'white'
          AND e.pub_year IS NOT NULL
        ORDER BY e.pub_year, e.event_id
    """

    by_year: dict[int, list[int]] = defaultdict(list)

    with get_connection(
        application_name="tier2-lance-stress",
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(query)

            for pub_year, event_id in cur.fetchall():
                by_year[int(pub_year)].append(int(event_id))

    return dict(sorted(by_year.items()))


def discover_tables(db):
    """Discover physical Lance tables through LanceDB's logical table API."""

    tables = []

    for table_name in db.list_tables().tables:
        match = TABLE_PATTERN.match(table_name)

        if match is None:
            raise RuntimeError(
                f"Unexpected Lance table name: {table_name!r}"
            )

        tables.append(
            {
                "name": table_name,
                "scale": match.group("scale"),
                "model": match.group("model"),
                "year_start": int(match.group("year_start")),
                "year_end": int(match.group("year_end")),
                "table": db.open_table(table_name),
            }
        )

    if not tables:
        raise RuntimeError("No Lance tables found")

    return tables


def tables_for_year(
    tables,
    *,
    scale: str,
    year: int,
):
    """Return the physical table covering a logical publication year."""

    matches = [
        entry
        for entry in tables
        if (
            entry["scale"] == scale
            and entry["year_start"] <= year <= entry["year_end"]
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {scale} table for year={year}; "
            f"found {len(matches)}: "
            f"{[entry['name'] for entry in matches]}"
        )

    return matches[0]


def reconstruct_vectors(
    table,
    event_ids: list[int],
) -> np.ndarray:
    """
    Reconstruct the seed vectors from Lance by stable event ID.

    The stress test deliberately avoids the normal observation-index wrapper.
    This keeps the experiment focused on native LanceDB multi-query search.
    """

    vectors: dict[int, np.ndarray] = {}

    for event_id in event_ids:
        rows = (
            table
            .search()
            .where(
                f"event_id = {int(event_id)}",
                prefilter=True,
            )
            .limit(1)
            .to_list()
        )

        if not rows:
            raise KeyError(
                f"Lance table {table.name!r} does not contain "
                f"event_id={event_id}"
            )

        returned_id = int(rows[0]["event_id"])

        if returned_id != event_id:
            raise RuntimeError(
                f"Expected event_id={event_id}, "
                f"got event_id={returned_id}"
            )

        vector = np.asarray(
            rows[0]["vector"],
            dtype=np.float32,
        )

        if vector.shape != (DIMENSIONS,):
            raise ValueError(
                f"Invalid vector shape for event_id={event_id}: "
                f"{vector.shape}"
            )

        vectors[event_id] = vector

    return np.asarray(
        [vectors[event_id] for event_id in event_ids],
        dtype=np.float32,
    )


def normalise_queries(
    queries: np.ndarray,
) -> np.ndarray:
    """Match the normalisation used by LanceObservationIndex."""

    norms = np.linalg.norm(
        queries,
        axis=1,
        keepdims=True,
    )

    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError(
            "Seed vectors contain a non-finite or zero norm"
        )

    return queries / norms


def search_year_scale(
    *,
    year: int,
    event_ids: list[int],
    entry,
) -> None:
    """Run every seed for one year as one native LanceDB multi-query request."""

    scale = entry["scale"]
    table = entry["table"]
    table_name = entry["name"]

    started = time.perf_counter()

    print(
        f"  {scale}: {len(event_ids)} simultaneous queries "
        f"against {table_name}",
        flush=True,
    )

    print(
        f"    reconstructing {len(event_ids)} seed vectors...",
        flush=True,
    )

    queries = reconstruct_vectors(
        table,
        event_ids,
    )

    queries = normalise_queries(queries)

    query_count = queries.shape[0]

    print(
        f"    executing native LanceDB search: "
        f"queries={query_count} "
        f"k={K} "
        f"search_k={K * OVERSAMPLE} "
        f"nprobes={NPROBES}",
        flush=True,
    )

    # query_index and _distance are generated by LanceDB's vector-search
    # result and are not physical table fields, so they cannot be included
    # in select(). The full result is deliberately retained for this stress
    # test so that native multi-query result semantics remain unchanged.
    request = (
        table
        .search(
            queries,
            vector_column_name="vector",
        )
        .nprobes(NPROBES)
        .limit(K * OVERSAMPLE)
        .where(
            f"year = {year}",
            prefilter=True,
        )
    )

    rows = request.to_list()

    print(
        f"    returned {len(rows)} rows",
        flush=True,
    )

    query_indexes = set()

    for row_index, row in enumerate(rows):
        if "query_index" not in row:
            raise RuntimeError(
                "\n"
                "LANCE FAILURE: missing query_index\n"
                f"  year={year}\n"
                f"  scale={scale}\n"
                f"  table={table_name}\n"
                f"  row_index={row_index}\n"
                f"  query_count={query_count}\n"
                f"  event_id={row.get('event_id')}\n"
                f"  row_year={row.get('year')}\n"
                f"  distance={row.get('_distance')}\n"
                f"  keys={list(row.keys())}"
            )

        query_index = int(row["query_index"])

        if not 0 <= query_index < query_count:
            raise RuntimeError(
                "\n"
                "LANCE FAILURE: invalid query_index\n"
                f"  year={year}\n"
                f"  scale={scale}\n"
                f"  table={table_name}\n"
                f"  row_index={row_index}\n"
                f"  query_index={query_index}\n"
                f"  query_count={query_count}\n"
                f"  event_id={row.get('event_id')}"
            )

        query_indexes.add(query_index)

    elapsed = time.perf_counter() - started

    print(
        f"  {scale}: OK "
        f"({len(query_indexes)}/{query_count} query indexes represented, "
        f"{elapsed:.2f}s)",
        flush=True,
    )


def main() -> int:
    print(
        f"Opening LanceDB: {LANCE_INDEXES_DIR}",
        flush=True,
    )

    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    print(
        "Discovering physical Lance tables...",
        flush=True,
    )

    tables = discover_tables(db)

    print(
        f"Found {len(tables)} physical Lance tables.",
        flush=True,
    )

    model_tables = [
        entry
        for entry in tables
        if entry["model"] == "macberth"
    ]

    if len(model_tables) != 30:
        raise RuntimeError(
            f"Expected 30 macberth tables, found {len(model_tables)}"
        )

    print(
        "Loading every WHITE seed event from PostgreSQL...",
        flush=True,
    )

    seeds_by_year = get_white_events_by_year()

    total_seeds = sum(
        len(event_ids)
        for event_ids in seeds_by_year.values()
    )

    print(
        f"WHITE seeds: {total_seeds}",
        flush=True,
    )

    print(
        f"Publication years: {len(seeds_by_year)}",
        flush=True,
    )

    print(
        "\nStarting full WHITE year-by-year native LanceDB stress test.",
        flush=True,
    )

    started = time.perf_counter()

    for year, event_ids in seeds_by_year.items():
        if year < 1656:
            continue

        year_started = time.perf_counter()

        print(
            f"\nYEAR {year}: {len(event_ids)} seed queries",
            flush=True,
        )

        for scale in ("local", "medium", "broad"):
            entry = tables_for_year(
                model_tables,
                scale=scale,
                year=year,
            )

            search_year_scale(
                year=year,
                event_ids=event_ids,
                entry=entry,
            )

        elapsed = time.perf_counter() - year_started

        print(
            f"YEAR {year}: COMPLETE in {elapsed:.2f}s",
            flush=True,
        )

    elapsed = time.perf_counter() - started

    print(
        "\nFULL TEST COMPLETE",
        flush=True,
    )

    print(
        f"Years tested: {len(seeds_by_year)}",
        flush=True,
    )

    print(
        f"Seed queries tested: {total_seeds}",
        flush=True,
    )

    print(
        f"Elapsed: {elapsed:.2f}s",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
