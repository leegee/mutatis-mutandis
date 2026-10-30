
from __future__ import annotations

import argparse
from collections import defaultdict

import lancedb
import pyarrow as pa

from lib.corpus_config import (
    LANCE_INDEXES_DIR,
    LANCE_BUCKET_SIZE,
    LANCE_MODEL_NAME,
)
from lib.corpus_db import get_connection
from tier1.vector_writer import year_bucket


CORPUS = "cme"
REPRESENTATIONS = ("l8", "last", "mean4")


def table_name(representation: str, year: int) -> str:
    start, end = year_bucket(year, LANCE_BUCKET_SIZE)
    return (
        f"passage_{representation}__{LANCE_MODEL_NAME}"
        f"__{start:04d}_{end:04d}"
    )


def load_authoritative_years() -> dict[str, int]:
    """Load CME document years from Postgres."""
    sql = """
        SELECT doc_id, pub_year
        FROM documents
        WHERE corpus = %s
          AND pub_year IS NOT NULL
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (CORPUS,))
            rows = cur.fetchall()

    years = {}
    for doc_id, year in rows:
        if doc_id in years and years[doc_id] != int(year):
            raise ValueError(
                f"Conflicting Postgres years for {doc_id!r}"
            )
        years[doc_id] = int(year)

    print(f"Loaded {len(years):,} authoritative CME document years")
    return years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes; otherwise only report them.",
    )
    args = parser.parse_args()

    years = load_authoritative_years()
    db = lancedb.connect(str(LANCE_INDEXES_DIR))
    existing_tables = set(db.list_tables().tables)

    totals = defaultdict(int)
    missing_docs = set()
    tables_to_process = [
        name
        for name in sorted(existing_tables)
        if name.startswith("passage_")
        and any(
            name.startswith(f"passage_{rep}__")
            for rep in REPRESENTATIONS
        )
    ]

    for source_name in tables_to_process:
        table = db.open_table(source_name)
        schema = table.schema
        if "corpus" not in schema.names or "doc_id" not in schema.names:
            continue
        if "pub_year" not in schema.names:
            continue

        # Read only CME rows. Fetch vectors only if relocation is needed.
        rows = (
            table.search()
            .where("corpus = 'cme'")
            .select([
                "passage_id", "corpus", "doc_id", "pub_year",
                "word_start", "word_end", "token_start", "token_end",
                "embedding_model",
            ])
            .limit(10_000_000)
            .to_list()
        )

        updates = []
        relocations = []

        for row in rows:
            doc_id = row["doc_id"]
            correct_year = years.get(doc_id)

            if correct_year is None:
                missing_docs.add(doc_id)
                continue

            old_year = row["pub_year"]
            if old_year == correct_year:
                continue

            target_name = table_name(
                source_name.split("__")[0].removeprefix("passage_"),
                correct_year,
            )

            if target_name == source_name:
                updates.append((row["passage_id"], correct_year))
            else:
                relocations.append(
                    (row, correct_year, target_name)
                )

        totals["checked"] += len(rows)
        totals["updates"] += len(updates)
        totals["relocations"] += len(relocations)

        print(
            f"{source_name}: {len(rows):,} CME rows; "
            f"{len(updates):,} year updates; "
            f"{len(relocations):,} bucket relocations"
        )

        if not args.apply:
            continue

        # In-place updates: pub_year is the only modified field.
        for passage_id, year in updates:
            table.update(
                where=f"passage_id = '{passage_id}'",
                values={"pub_year": year},
            )

        # Relocate rows only when the corrected year crosses a bucket.
        # Fetch vectors separately so they remain unchanged.
        for row, correct_year, target_name in relocations:
            full = (
                table.search()
                .where(f"passage_id = '{row['passage_id']}'")
                .limit(1)
                .to_list()
            )
            if not full:
                raise RuntimeError(
                    f"Passage disappeared: {row['passage_id']}"
                )

            full_row = full[0]
            vector = full_row["vector"]

            new_row = dict(full_row)
            new_row["pub_year"] = correct_year

            if target_name not in set(db.list_tables().tables):
                target = db.create_table(
                    target_name,
                    schema=table.schema,
                )
            else:
                target = db.open_table(target_name)

            # Insert before deleting the original, to avoid data loss.
            target.add([new_row])
            table.delete(
                f"passage_id = '{row['passage_id']}'"
            )

            totals["moved"] += 1

    print("\nSummary")
    print(f"Rows checked:       {totals['checked']:,}")
    print(f"Year updates:       {totals['updates']:,}")
    print(f"Bucket relocations: {totals['relocations']:,}")

    if missing_docs:
        print(
            f"Missing Postgres matches: {len(missing_docs):,}"
        )
        for doc_id in sorted(missing_docs)[:30]:
            print(f"  {doc_id}")

    if args.apply:
        print(f"Rows relocated:     {totals['moved']:,}")
        print("Changes applied.")
    else:
        print("Dry run only. Re-run with --apply to make changes.")


if __name__ == "__main__":
    main()
