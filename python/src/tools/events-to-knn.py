#!/usr/bin/env python
"""
Compare cross-document local-vector neighbours of five selected church events.

Invariants:
- Search local embeddings only; do not fuse scales or use RRF.
- Rank by the raw similarities returned by FAISS.
- Exclude the target event and every event from its source (corpus, doc_id).
- Resolve indexed-event provenance from the complete Zarr event lookup.
- Retrieve surrounding text and document metadata from PostgreSQL.
- Database access is read-only; FAISS indexes and Zarr stores are not modified.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from lib.corpus_config import ZARR_PATH
from lib.corpus_db import get_connection
from lib.eebo_faiss import EeboFaissIndex
from lib.zarr_event_lookup import ZarrEventLookup


TARGETS = {
    8465373009736818620: "A57456:1272 — theological community",
    2882461281543078623: "A57456:1455 — treasury and social provision",
    3180480654897567141: "A11764:7859 — episcopal governance",
    7231107858140400046: "A02549:3233 — anti-episcopacy polemic",
    8310090952793042223: "A00089:386 — church-state relations",
}


def get_event_metadata(lookup: ZarrEventLookup, event_id: int) -> dict:
    """Resolve event provenance without reconstructing its embedding."""
    try:
        return lookup.get_event_metadata(int(event_id))
    except KeyError as exc:
        raise RuntimeError(
            f"Event {event_id} is absent from the Zarr event lookup"
        ) from exc


def build_local_index_lookup(indices) -> dict:
    """Map each local-index event ID to its year, index, and partition IDs."""
    lookup = {}

    for year, scales in indices.items():
        index = scales.get("local")
        if index is None:
            continue

        ids = set(map(int, index.ids()))

        for event_id in ids:
            if event_id in lookup:
                raise RuntimeError(
                    f"Event {event_id} occurs in multiple local FAISS partitions"
                )
            lookup[event_id] = (year, index, ids)

    return lookup


def build_event_document_lookup(
    index_lookup: dict,
    lookup: ZarrEventLookup,
) -> dict[int, tuple[str, str]]:
    """Validate FAISS coverage and map indexed IDs to (corpus, doc_id)."""
    event_doc_keys = {}
    missing = []

    for event_id in index_lookup:
        try:
            pos = lookup.get_pos(event_id)
        except KeyError:
            missing.append(event_id)
            continue

        event_doc_keys[event_id] = (
            str(lookup.corpus[pos]),
            str(lookup.doc_id[pos]),
        )

    if missing:
        raise RuntimeError(
            f"{len(missing):,} FAISS events are absent from Zarr; "
            f"sample IDs: {missing[:10]}. "
            "Check that the FAISS indexes and Zarr stores belong to "
            "the same corpus build."
        )

    return event_doc_keys


def nearest_local_neighbours(
    target_id: int,
    target_doc_key: tuple[str, str],
    index_lookup: dict,
    event_doc_keys: dict[int, tuple[str, str]],
    top_k: int,
) -> tuple[object | None, list[dict]]:
    """
    Merge cross-document local-index results across partitions.

    Each partition is searched deeply enough to obtain its top K eligible
    results, or until that partition is exhausted. This prevents events
    from the target document from consuming the result limit.
    """
    target_entry = index_lookup.get(target_id)
    if target_entry is None:
        return None, []

    target_year, target_index, _ = target_entry
    query_vector = target_index.reconstruct(target_id)
    candidates = {}

    # Partition IDs repeat for every event in a partition.
    searched_indexes = set()

    for year, index, ids in index_lookup.values():
        index_key = id(index)
        if index_key in searched_indexes:
            continue
        searched_indexes.add(index_key)

        if not ids:
            continue

        k = min(max(top_k + 1, 16), len(ids))

        while True:
            similarities, neighbour_ids = index.search(query_vector, k)
            eligible = {}

            for score, raw_neighbour_id in zip(
                similarities[0],
                neighbour_ids[0],
            ):
                neighbour_id = int(raw_neighbour_id)

                if neighbour_id < 0 or neighbour_id == target_id:
                    continue

                neighbour_doc_key = event_doc_keys.get(neighbour_id)
                if neighbour_doc_key is None:
                    raise RuntimeError(
                        f"FAISS neighbour {neighbour_id} has no "
                        "validated Zarr document metadata"
                    )

                if neighbour_doc_key == target_doc_key:
                    continue

                score = float(score)
                previous = eligible.get(neighbour_id)

                if previous is None or score > previous["similarity"]:
                    eligible[neighbour_id] = {
                        "neighbour_event_id": neighbour_id,
                        "similarity": score,
                        "neighbour_index_year": year,
                    }

            if len(eligible) >= top_k or k >= len(ids):
                break

            k = min(k * 2, len(ids))

        for neighbour_id, result in eligible.items():
            previous = candidates.get(neighbour_id)

            if (
                previous is None
                or result["similarity"] > previous["similarity"]
            ):
                candidates[neighbour_id] = result

    ranked = sorted(
        candidates.values(),
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return target_year, ranked[:top_k]


def fetch_contexts(
    positions: set[tuple[str, int]],
    radius: int = 12,
) -> dict[tuple[str, int], dict]:
    """Fetch token windows and document metadata from PostgreSQL."""
    contexts = {}
    if not positions:
        return contexts

    conn = get_connection()

    try:
        with conn.cursor() as cur:
            for doc_id, token_idx in positions:
                cur.execute(
                    """
                    SELECT
                        t.token_idx,
                        t.token,
                        d.pub_year,
                        d.title,
                        d.author,
                        d.pub_place
                    FROM tokens t
                    JOIN documents d ON d.doc_id = t.doc_id
                    WHERE t.doc_id = %s
                      AND t.token_idx BETWEEN %s AND %s
                    ORDER BY t.token_idx
                    """,
                    (
                        doc_id,
                        max(0, token_idx - radius),
                        token_idx + radius,
                    ),
                )
                rows = cur.fetchall()

                if not rows:
                    contexts[(doc_id, token_idx)] = {
                        "windowText": "",
                        "pub_year": None,
                        "title": None,
                        "author": None,
                        "pub_place": None,
                    }
                    continue

                tokens = []

                for row in rows:
                    current_idx, token = int(row[0]), str(row[1])
                    if current_idx == token_idx:
                        token = f"⟦{token}⟧"
                    tokens.append(token)

                contexts[(doc_id, token_idx)] = {
                    "windowText": " ".join(tokens),
                    "pub_year": rows[0][2],
                    "title": rows[0][3],
                    "author": rows[0][4],
                    "pub_place": rows[0][5],
                }
    finally:
        conn.close()

    return contexts


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect cross-document local-vector neighbours "
            "of selected church events."
        )
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument(
        "--output",
        default="church_local_cross_document_neighbours.csv",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.radius < 0:
        parser.error("--radius cannot be negative")

    print("Loading local FAISS indexes...")
    indices = EeboFaissIndex.load_all()

    print("Building event-ID lookup...")
    index_lookup = build_local_index_lookup(indices)
    print(f"Local-index events: {len(index_lookup):,}")

    print("Loading Zarr event metadata...")
    lookup = ZarrEventLookup(ZARR_PATH)
    print(f"Zarr events: {len(lookup):,}")

    print("Validating FAISS event coverage and resolving documents...")
    event_doc_keys = build_event_document_lookup(index_lookup, lookup)
    print(f"Indexed events resolved in Zarr: {len(event_doc_keys):,}")

    target_positions = {}

    for event_id, description in TARGETS.items():
        try:
            target_positions[event_id] = get_event_metadata(
                lookup,
                event_id,
            )
        except RuntimeError:
            print(f"Target missing from Zarr: {event_id} — {description}")

    missing_targets = set(TARGETS) - target_positions.keys()
    if missing_targets:
        raise RuntimeError(
            f"{len(missing_targets)} target events are missing from Zarr: "
            f"{sorted(missing_targets)}"
        )

    print("Target document checks:")
    for event_id, description in TARGETS.items():
        position = target_positions[event_id]
        print(
            f"  {event_id}: corpus={position['corpus']!r}, "
            f"doc_id={position['doc_id']!r}, "
            f"token_idx={position['token_idx']}"
        )

    results = []
    all_ids = set()

    for target_id, description in TARGETS.items():
        target_pos = target_positions[target_id]
        target_doc_key = (
            str(target_pos["corpus"]),
            str(target_pos["doc_id"]),
        )

        target_year, neighbours = nearest_local_neighbours(
            target_id=target_id,
            target_doc_key=target_doc_key,
            index_lookup=index_lookup,
            event_doc_keys=event_doc_keys,
            top_k=args.top_k,
        )

        if target_year is None:
            print(f"Target absent from local FAISS: {description}")
            continue

        print(f"{description}: {len(neighbours)} cross-document neighbours")

        for rank, neighbour in enumerate(neighbours, start=1):
            results.append({
                "target_event_id": target_id,
                "target_description": description,
                "target_index_year": target_year,
                "rank": rank,
                **neighbour,
            })
            all_ids.add(neighbour["neighbour_event_id"])

    print("Resolving target and neighbour metadata from Zarr...")
    needed_ids = all_ids | set(target_positions)

    positions_by_id = {
        event_id: get_event_metadata(lookup, event_id)
        for event_id in needed_ids
    }

    # PostgreSQL context retrieval is keyed by doc_id; the corpus remains
    # in the exported provenance so cross-corpus identities stay distinguishable.
    all_positions = {
        (str(row["doc_id"]), int(row["token_idx"]))
        for row in positions_by_id.values()
    }

    print("Fetching contexts from PostgreSQL...")
    contexts = fetch_contexts(all_positions, radius=args.radius)

    fields = [
        "target_event_id",
        "target_description",
        "target_corpus",
        "target_doc_id",
        "target_token_idx",
        "target_year",
        "target_title",
        "target_author",
        "target_windowText",
        "target_index_year",
        "rank",
        "neighbour_event_id",
        "similarity",
        "neighbour_index_year",
        "neighbour_corpus",
        "neighbour_doc_id",
        "neighbour_token_idx",
        "neighbour_year",
        "neighbour_title",
        "neighbour_author",
        "neighbour_pub_place",
        "neighbour_windowText",
    ]

    output_path = Path(args.output)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for result in results:
            target_pos = positions_by_id.get(result["target_event_id"])
            neighbour_pos = positions_by_id.get(
                result["neighbour_event_id"]
            )

            target_context = (
                contexts.get(
                    (
                        str(target_pos["doc_id"]),
                        int(target_pos["token_idx"]),
                    ),
                    {},
                )
                if target_pos
                else {}
            )

            neighbour_context = (
                contexts.get(
                    (
                        str(neighbour_pos["doc_id"]),
                        int(neighbour_pos["token_idx"]),
                    ),
                    {},
                )
                if neighbour_pos
                else {}
            )

            writer.writerow({
                "target_event_id": result["target_event_id"],
                "target_description": result["target_description"],
                "target_corpus": (
                    target_pos["corpus"] if target_pos else ""
                ),
                "target_doc_id": (
                    target_pos["doc_id"] if target_pos else ""
                ),
                "target_token_idx": (
                    target_pos["token_idx"] if target_pos else ""
                ),
                "target_year": target_context.get("pub_year"),
                "target_title": target_context.get("title"),
                "target_author": target_context.get("author"),
                "target_windowText": target_context.get("windowText", ""),
                "target_index_year": result["target_index_year"],
                "rank": result["rank"],
                "neighbour_event_id": result["neighbour_event_id"],
                "similarity": result["similarity"],
                "neighbour_index_year": result["neighbour_index_year"],
                "neighbour_corpus": (
                    neighbour_pos["corpus"] if neighbour_pos else ""
                ),
                "neighbour_doc_id": (
                    neighbour_pos["doc_id"] if neighbour_pos else ""
                ),
                "neighbour_token_idx": (
                    neighbour_pos["token_idx"] if neighbour_pos else ""
                ),
                "neighbour_year": neighbour_context.get("pub_year"),
                "neighbour_title": neighbour_context.get("title"),
                "neighbour_author": neighbour_context.get("author"),
                "neighbour_pub_place": neighbour_context.get("pub_place"),
                "neighbour_windowText": neighbour_context.get(
                    "windowText",
                    "",
                ),
            })

    print(f"Wrote {len(results)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()