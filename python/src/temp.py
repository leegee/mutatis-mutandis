
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import lancedb

from lib.corpus_config import LANCE_INDEXES_DIR


DOCS_CSV = Path("temp_albino_docs.csv")


def main() -> None:
    with DOCS_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        requested = {
            (row["corpus"], row["doc_id"])
            for row in reader
        }

    if not requested:
        raise RuntimeError(f"No document IDs found in {DOCS_CSV}")

    db = lancedb.connect(str(LANCE_INDEXES_DIR))
    tables = sorted(
        name for name in db.list_tables().tables
        if name.startswith("passage_")
        and "__macberth__" in name
    )

    found: dict[tuple[str, str], list[dict]] = defaultdict(list)

    print(f"Index: {LANCE_INDEXES_DIR}")
    print(f"Documents to check: {len(requested)}")
    print(f"Passage tables: {len(tables)}")

    for name in tables:
        table = db.open_table(name)
        arrow = table.search().select([
            "corpus",
            "doc_id",
            "pub_year",
            "token_start",
            "token_end",
            "passage_id",
        ]).to_arrow()

        matches = [
            row
            for row in arrow.to_pylist()
            if (row["corpus"], row["doc_id"]) in requested
        ]

        for row in matches:
            found[(row["corpus"], row["doc_id"])].append({
                **row,
                "table": name,
            })

        print(f"{name}: {len(matches)} matching passages")

    present = set(found)
    missing = sorted(requested - present)

    print(f"\nDocuments present in Lance: {len(present)}")
    print(f"Documents absent from Lance: {len(missing)}")

    for corpus, doc_id in sorted(requested):
        rows = found.get((corpus, doc_id), [])
        if rows:
            print(f"\nPRESENT {corpus}:{doc_id} ({len(rows)} passages)")
            for row in rows[:5]:
                print(
                    f"  {row['table']} "
                    f"year={row['pub_year']} "
                    f"tokens={row['token_start']}:{row['token_end']} "
                    f"id={row['passage_id']}"
                )
            if len(rows) > 5:
                print(f"  ... and {len(rows) - 5} more")
        else:
            print(f"\nMISSING {corpus}:{doc_id}")


if __name__ == "__main__":
    main()
