# lib/repair_neighbours.py
from __future__ import annotations

from collections import defaultdict

import numpy as np

from lib.eebo_logging import logger

# Sentinels / constants shared with tier2
_NO_WPOS = -1


def find_events_without_neighbours(con, batch_size: int = 5000):
    """
    Yield batches of (event_id, pub_year) for events that have no
    outgoing neighbour rows.
    """
    rows = con.execute(
        """
        SELECT e.event_id, e.pub_year
        FROM events e
        LEFT JOIN neighbours n
               ON n.event_id = e.event_id
        WHERE n.event_id IS NULL
          AND e.pub_year IS NOT NULL
        ORDER BY e.pub_year, e.event_id
        """
    ).fetchall()

    logger.info(f"[tier2] {len(rows):,} events require neighbour repair")

    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def backfill_neighbours_for_year(
    con,
    lookup,
    indexes,
    year: int,
    event_ids: list[int],
    top_n: int,
    rrf_k: int,
    oversample: int,
) -> int:
    """
    Compute and insert depth-1 neighbours for a batch of events
    that all share the same publication year.

    Returns the number of neighbour rows written.
    """
    from lib.eebo_faiss import multiscale_search
    # ensure_events lives in the tier2 module; import locally to avoid cycles
    from tier2_0_concept_events import ensure_events

    if not event_ids:
        return 0

    positions = []
    valid_ids = []
    for eid in event_ids:
        try:
            positions.append(lookup.get_pos(eid))
            valid_ids.append(eid)
        except KeyError:
            logger.warning(f"[tier2] event {eid} missing from lookup; skipping")

    if not positions:
        return 0

    positions = np.asarray(positions, dtype=np.int64)

    results = multiscale_search(
        indexes,
        lookup,
        positions,
        top_n,
        pub_year=year,
        rrf_k=rrf_k,
        oversample=oversample,
    )

    all_neighbour_ids: set[int] = set()
    neighbour_rows = []

    for eid, neighbours in zip(valid_ids, results):
        for item in neighbours:
            nid = item["event_id"]
            if nid == eid:
                continue
            all_neighbour_ids.add(nid)

            try:
                npos = lookup.get_pos(nid)
            except KeyError:
                continue

            wpos = int(lookup.window_token_pos[npos])
            neighbour_rows.append((
                eid,
                nid,
                1,                                      # depth
                None,                                   # via_event_id
                int(lookup.vector_id[npos]),
                str(lookup.token[npos]),
                str(lookup.doc_id[npos]),
                int(lookup.pub_year[npos]),
                int(lookup.token_idx[npos]),
                int(lookup.window_id[npos]),
                None if wpos == _NO_WPOS else wpos,
                item["rrf_score"],
                item.get("score_local"),
                item.get("score_medium"),
                item.get("score_broad"),
            ))

    # Ensure both the source events and any newly-seen neighbours exist
    ensure_events(con, lookup, set(valid_ids) | all_neighbour_ids)

    if neighbour_rows:
        con.executemany(
            """
            INSERT OR IGNORE INTO neighbours (
                event_id, neighbour_event_id, depth, via_event_id,
                vector_id, token, doc_id, pub_year, token_idx,
                window_id, window_token_pos, score,
                score_local, score_medium, score_broad
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            neighbour_rows,
        )

    return len(neighbour_rows)


def repair_missing_neighbours(
    *,
    con,
    lookup,
    indexes,
    top_n: int,
    rrf_k: int,
    oversample: int,
    batch_size: int = 5000,
):
    """
    Database maintenance.

    Finds every event lacking outgoing neighbours and computes them.
    Safe to run repeatedly.
    """
    total_events = 0
    total_rows = 0
    skipped = 0

    for batch in find_events_without_neighbours(con, batch_size):
        by_year: dict[int, list[int]] = defaultdict(list)

        for event_id, year in batch:
            by_year[int(year)].append(event_id)

        for year, event_ids in by_year.items():
            if year not in indexes:
                logger.warning(
                    f"[tier2] no index for year {year}; skipping {len(event_ids)} events"
                )
                skipped += len(event_ids)
                continue

            inserted = backfill_neighbours_for_year(
                con,
                lookup,
                indexes,
                year,
                event_ids,
                top_n,
                rrf_k,
                oversample,
            )

            total_events += len(event_ids)
            total_rows += inserted

        con.commit()
        logger.info(
            f"[tier2] repaired {total_events:,} events ({total_rows:,} neighbour rows)"
        )

    logger.info(
        f"[tier2] neighbour repair complete "
        f"({total_events:,} events, "
        f"{total_rows:,} rows, "
        f"{skipped:,} skipped)"
    )
