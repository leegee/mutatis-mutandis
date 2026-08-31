#!/usr/bin/env python

"""
tier1/test_parquet_integrity.py - Check the integrity of a Tier 1 Parquet
observation store against PostgreSQL.

This test validates the observation identity/metadata layer without loading
embedding vectors. A corpus token may legitimately have multiple observations
because contextual windows overlap, so the comparison is performed on distinct
(corpus, doc_id, token_idx) occurrences.

The test assumes the store contains unmasked Tier 1 observations. Masked Tier 1
has a deliberately different observation population and therefore requires a
different completeness rule.

For a sharded Tier 1 run, each Parquet shard is checked against the corresponding
PostgreSQL document shard using the same hash expression as Tier 1.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

from lib.corpus_db import get_connection
from lib.corpus_logging import logger
from lib.stopwords_min import STOPWORDS

from tier1.observation_store_api import (
    resolve_store_path,
)


def is_content_token(token: str) -> bool:
    stripped = token.strip().lower()

    if not stripped or stripped in STOPWORDS:
        return False

    if all(
        unicodedata.category(c).startswith(("P", "S", "Z"))
        for c in stripped
    ):
        return False

    return True


def resolve_integrity_stores(args) -> list[Path]:
    """
    Resolve the Parquet stores participating in this integrity check.

    A sharded Tier 1 run produces one independent store per shard. The
    integrity test must inspect those stores rather than silently falling
    back to the unsharded default path.
    """
    if args.store:
        store = Path(args.store)

        if args.num_shards > 1:
            return [
                store.parent / f"{store.name}{shard}"
                for shard in range(args.num_shards)
            ]

        return [store]

    if args.num_shards > 1:
        return [
            resolve_store_path(
                store_backend="parquet",
                masked=False,
                store=None,
                shard=shard,
                num_shards=args.num_shards,
            )
            for shard in range(args.num_shards)
        ]

    return [
        resolve_store_path(
            store_backend="parquet",
            masked=False,
            store=None,
            shard=None,
            num_shards=1,
        )
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check Tier 1 Parquet output against PostgreSQL"
    )

    parser.add_argument(
        "--store",
        type=str,
        default=None,
        help="Override Parquet store path; otherwise use the configured default",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of Tier 1 shards",
    )

    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="Only test this many documents per shard (useful for development)",
    )

    return parser.parse_args()


def load_parquet_metadata(store_path: Path):
    dataset = ds.dataset(
        str(store_path),
        format="parquet",
        partitioning="hive",
    )

    required_columns = {
        "event_id",
        "vector_id",
        "doc_id",
        "corpus",
        "pub_year",
        "token_idx",
        "token",
        "window_id",
        "window_token_pos",
    }

    actual_columns = set(dataset.schema.names)
    missing = required_columns - actual_columns

    if missing:
        raise AssertionError(
            f"Parquet schema is missing required columns: {sorted(missing)}"
        )

    table = dataset.to_table(
        columns=sorted(required_columns),
    )

    rows = table.to_pylist()

    # Keep observations grouped by token occurrence. Multiple observations
    # for one occurrence are expected because contextual windows overlap.
    observations = defaultdict(list)

    for row in rows:
        key = (
            row["corpus"],
            row["doc_id"],
            int(row["token_idx"]),
        )
        observations[key].append(row)

    return observations, len(rows)


def load_db_tokens(
    conn,
    limit_docs: int | None,
    *,
    shard: int | None = None,
    num_shards: int = 1,
):
    """
    Load the PostgreSQL token metadata for one Tier 1 shard.

    The shard expression deliberately matches CorpusProcessor._shard_clause()
    in tier1_corpus2zarr.py:

        abs(hashtext(corpus || ':' || doc_id)) % num_shards = shard

    Keeping this expression identical is an invariant: otherwise the integrity
    test could report false missing/unexpected observations.
    """
    cur = conn.cursor()

    params = []
    shard_clause = ""

    if shard is not None and num_shards > 1:
        shard_clause = """
            AND abs(hashtext(corpus || ':' || doc_id)) %% %s = %s
        """
        params.extend([num_shards, shard])

    sql = f"""
        SELECT
            corpus,
            doc_id,
            token_idx,
            vector_id,
            token,
            pub_year
        FROM pamphlet_tokens
        WHERE 1 = 1
        {shard_clause}
        ORDER BY corpus, doc_id, token_idx
    """

    cur.execute(sql, params)

    expected = {}
    documents = set()

    for corpus, doc_id, token_idx, vector_id, token, pub_year in cur:
        key = (corpus, doc_id, int(token_idx))

        if key not in expected:
            expected[key] = {
                "vector_id": int(vector_id),
                "token": token,
                "pub_year": int(pub_year),
            }
            documents.add((corpus, doc_id))

    cur.close()

    if limit_docs is not None:
        selected_docs = sorted(documents)[:limit_docs]
        selected = set(selected_docs)

        expected = {
            key: value
            for key, value in expected.items()
            if (key[0], key[1]) in selected
        }

    return expected


def check_shard(
    conn,
    store_path: Path,
    *,
    shard: int,
    num_shards: int,
    limit_docs: int | None,
):
    logger.info("-" * 70)
    logger.info(f"Checking shard {shard}: {store_path}")

    observations, parquet_row_count = load_parquet_metadata(store_path)

    logger.info(
        f"Parquet observation rows: {parquet_row_count:,}"
    )
    logger.info(
        f"Distinct token occurrences: {len(observations):,}"
    )

    db_tokens = load_db_tokens(
        conn,
        limit_docs,
        shard=shard if num_shards > 1 else None,
        num_shards=num_shards,
    )

    expected_content = {
        key: value
        for key, value in db_tokens.items()
        if is_content_token(value["token"])
    }

    logger.info(
        f"Expected DB content tokens: {len(expected_content):,}"
    )

    failures = []

    missing = sorted(
        set(expected_content) - set(observations)
    )

    unexpected = sorted(
        set(observations) - set(db_tokens)
    )

    if missing:
        failures.append(
            f"{len(missing):,} expected content-token occurrences "
            f"are missing from Parquet"
        )

    if unexpected:
        failures.append(
            f"{len(unexpected):,} Parquet token occurrences do not "
            f"exist in PostgreSQL"
        )

    metadata_mismatches = []

    for key, rows in observations.items():
        db_row = db_tokens.get(key)

        if db_row is None:
            continue

        expected_vector_id = db_row["vector_id"]
        expected_token = db_row["token"]
        expected_year = db_row["pub_year"]

        for row in rows:
            if int(row["vector_id"]) != expected_vector_id:
                metadata_mismatches.append(
                    (
                        key,
                        "vector_id",
                        expected_vector_id,
                        row["vector_id"],
                    )
                )

            if row["token"] != expected_token:
                metadata_mismatches.append(
                    (
                        key,
                        "token",
                        expected_token,
                        row["token"],
                    )
                )

            if int(row["pub_year"]) != expected_year:
                metadata_mismatches.append(
                    (
                        key,
                        "pub_year",
                        expected_year,
                        row["pub_year"],
                    )
                )

    if metadata_mismatches:
        failures.append(
            f"{len(metadata_mismatches):,} Parquet rows have "
            f"metadata that disagrees with PostgreSQL"
        )

    non_content = []

    for key in observations:
        db_row = db_tokens.get(key)

        if db_row is not None and not is_content_token(db_row["token"]):
            non_content.append(key)

    if non_content:
        failures.append(
            f"{len(non_content):,} Parquet token occurrences are "
            f"not content tokens"
        )

    # These checks catch malformed observations before downstream indexing.
    invalid_event_ids = 0
    invalid_window_positions = 0

    for rows in observations.values():
        for row in rows:
            if row["event_id"] is None:
                invalid_event_ids += 1

            if (
                row["window_id"] is None
                or row["window_token_pos"] is None
            ):
                invalid_window_positions += 1

    if invalid_event_ids:
        failures.append(
            f"{invalid_event_ids:,} rows have NULL event_id"
        )

    if invalid_window_positions:
        failures.append(
            f"{invalid_window_positions:,} rows have NULL window metadata"
        )

    logger.info("")
    logger.info(f"Shard {shard} results")
    logger.info("-----------------")
    logger.info(
        f"DB content tokens           : {len(expected_content):,}"
    )
    logger.info(
        f"Parquet distinct tokens     : {len(observations):,}"
    )
    logger.info(
        f"Parquet observation rows    : {parquet_row_count:,}"
    )
    logger.info(
        f"Missing content tokens      : {len(missing):,}"
    )
    logger.info(
        f"Unexpected token occurrences: {len(unexpected):,}"
    )
    logger.info(
        f"Metadata mismatches         : {len(metadata_mismatches):,}"
    )
    logger.info(
        f"Non-content observations    : {len(non_content):,}"
    )

    if failures:
        logger.info("")
        logger.info("INTEGRITY CHECK FAILED")

        for failure in failures:
            logger.info(f"  - {failure}")

        if missing:
            logger.info("")
            logger.info("First missing occurrences:")
            for key in missing[:10]:
                logger.info(f"  {key}")

        if unexpected:
            logger.info("")
            logger.info("First unexpected occurrences:")
            for key in unexpected[:10]:
                logger.info(f"  {key}")

        if metadata_mismatches:
            logger.info("")
            logger.info("First metadata mismatches:")
            for mismatch in metadata_mismatches[:10]:
                logger.info(f"  {mismatch}")

        return failures

    logger.info("")
    logger.info("INTEGRITY CHECK PASSED")

    return []


def main():
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    store_paths = resolve_integrity_stores(args)

    conn = get_connection()

    try:
        all_failures = []

        for shard, store_path in enumerate(store_paths):
            if not store_path.exists():
                failure = f"shard {shard}: store does not exist: {store_path}"
                logger.error(failure)
                all_failures.append(failure)
                continue

            failures = check_shard(
                conn,
                store_path,
                shard=shard,
                num_shards=args.num_shards,
                limit_docs=args.limit_docs,
            )

            all_failures.extend(
                f"shard {shard}: {failure}"
                for failure in failures
            )

        logger.info("")
        logger.info("==============================")

        if all_failures:
            logger.info("OVERALL INTEGRITY CHECK FAILED")

            for failure in all_failures:
                logger.info(f"  - {failure}")

            return 1

        logger.info("OVERALL INTEGRITY CHECK PASSED")
        return 0

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
