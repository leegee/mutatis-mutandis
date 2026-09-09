# tier2/inspect_run.py

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from lib.corpus_config import CORPUS_TIER2_DB_PATH
from lib.corpus_db import analysis_db_connection, get_connection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Tier 2 neighbour edges with PostgreSQL metadata."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=CORPUS_TIER2_DB_PATH,
        help="Path to the Tier 2 SQLite database.",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Tier 2 retrieval run to inspect.",
    )
    parser.add_argument(
        "--per-seed",
        type=int,
        default=10,
        help="Maximum neighbours to show for each seed.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show an aggregate neighbour-token summary instead of edges.",
    )
    return parser.parse_args()


def load_edges(db_path: Path, run_id: int) -> list[dict]:
    if not db_path.exists():
        raise FileNotFoundError(
            f"Tier 2 database does not exist: {db_path}"
        )

    with analysis_db_connection(str(db_path)) as con:
        tables = {
            row[0]
            for row in con.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

        if "neighbour_edges" not in tables:
            raise RuntimeError(
                f"Database does not contain the Tier 2 schema: {db_path}\n"
                f"Found tables: {', '.join(sorted(tables))}"
            )

        rows = con.execute(
            """
            SELECT
                seed_event_id,
                neighbour_event_id,
                depth,
                via_event_id,
                rank,
                score,
                score_local,
                score_medium,
                score_broad
            FROM neighbour_edges
            WHERE run_id = ?
            ORDER BY seed_event_id, rank
            """,
            (run_id,),
        ).fetchall()

    columns = (
        "seed_event_id",
        "neighbour_event_id",
        "depth",
        "via_event_id",
        "rank",
        "score",
        "score_local",
        "score_medium",
        "score_broad",
    )

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def load_event_metadata(
    pg_conn,
    event_ids: set[int],
) -> dict[int, dict]:
    if not event_ids:
        return {}

    ids = list(event_ids)

    with pg_conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                e.event_id,
                e.corpus,
                e.doc_id,
                e.token_idx,
                e.token,
                e.pub_year,
                t.raw_token,
                t.canonical,
                d.title,
                d.author
            FROM events e
            JOIN tokens t
              ON t.doc_id = e.doc_id
             AND t.token_idx = e.token_idx
            JOIN documents d
              ON d.doc_id = e.doc_id
            WHERE e.event_id = ANY(%s)
            """,
            (ids,),
        )

        rows = cur.fetchall()

    columns = (
        "event_id",
        "corpus",
        "doc_id",
        "token_idx",
        "token",
        "pub_year",
        "raw_token",
        "canonical",
        "title",
        "author",
    )

    metadata = {
        row[0]: dict(zip(columns, row))
        for row in rows
    }

    missing = event_ids - metadata.keys()

    if missing:
        raise RuntimeError(
            f"{len(missing)} event IDs from Tier 2 are missing "
            f"from PostgreSQL: {sorted(missing)[:10]}"
        )

    return metadata


def print_edge_inspection(
    edges: list[dict],
    metadata: dict[int, dict],
    per_seed: int,
) -> None:
    grouped: dict[int, list[dict]] = defaultdict(list)

    for edge in edges:
        grouped[edge["seed_event_id"]].append(edge)

    for seed_event_id, seed_edges in grouped.items():
        seed = metadata[seed_event_id]

        print()
        print(
            f'SEED {seed_event_id}: '
            f'{seed["token"]!r} '
            f'year={seed["pub_year"]} '
            f'doc={seed["doc_id"]} '
            f'token_idx={seed["token_idx"]}'
        )

        if seed["title"]:
            print(f'  title: {seed["title"]}')
        if seed["author"]:
            print(f'  author: {seed["author"]}')

        for edge in seed_edges[:per_seed]:
            neighbour = metadata[edge["neighbour_event_id"]]

            print(
                f'  {edge["rank"]:>2} '
                f'{neighbour["event_id"]} '
                f'{neighbour["token"]!r:<20} '
                f'year={neighbour["pub_year"]} '
                f'rrf={edge["score"]:.5f} '
                f'local={_format_score(edge["score_local"])} '
                f'medium={_format_score(edge["score_medium"])} '
                f'broad={_format_score(edge["score_broad"])} '
                f'doc={neighbour["doc_id"]}'
            )


def print_token_summary(
    edges: list[dict],
    metadata: dict[int, dict],
) -> None:
    counts: dict[str, int] = defaultdict(int)
    seed_counts: dict[str, set[int]] = defaultdict(set)

    for edge in edges:
        neighbour = metadata[edge["neighbour_event_id"]]
        token = neighbour["token"]

        counts[token] += 1
        seed_counts[token].add(edge["seed_event_id"])

    print()
    print("NEIGHBOUR TOKEN SUMMARY")
    print()
    print(f'{"token":<25} {"edges":>8} {"seeds":>8}')
    print("-" * 45)

    for token, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0].lower()),
    ):
        print(
            f'{token:<25} '
            f'{count:>8} '
            f'{len(seed_counts[token]):>8}'
        )


def _format_score(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.5f}"


def main() -> None:
    args = parse_args()

    if args.per_seed <= 0:
        raise ValueError("--per-seed must be positive")

    if not args.db_path.exists():
        raise FileNotFoundError(args.db_path)

    edges = load_edges(args.db_path, args.run_id)

    if not edges:
        print(f"No neighbour edges found for run {args.run_id}.")
        return

    event_ids = {
        edge["seed_event_id"]
        for edge in edges
    } | {
        edge["neighbour_event_id"]
        for edge in edges
    }

    with get_connection(
        application_name="tier2-inspect",
    ) as pg_conn:
        metadata = load_event_metadata(
            pg_conn,
            event_ids,
        )

    print(
        f"run={args.run_id} "
        f"edges={len(edges)} "
        f"seeds={len({e['seed_event_id'] for e in edges})} "
        f"events={len(metadata)}"
    )

    if args.summary:
        print_token_summary(edges, metadata)
    else:
        print_edge_inspection(
            edges,
            metadata,
            args.per_seed,
        )


if __name__ == "__main__":
    main()
