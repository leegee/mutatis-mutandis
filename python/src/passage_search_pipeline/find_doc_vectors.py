
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lancedb
import numpy as np

from lib.corpus_config import LANCE_INDEXES_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find existing LanceDB passage vectors by document ID."
    )
    parser.add_argument(
        "doc_ids",
        nargs="+",
        help="Document IDs to find, e.g. CLMET3255 A12345",
    )
    parser.add_argument(
        "--representation",
        choices=["l8", "last", "mean4", "all"],
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/export/doc_vectors"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db = lancedb.connect(str(LANCE_INDEXES_DIR))

    requested = set(args.doc_ids)
    representations = (
        ["l8", "last", "mean4"]
        if args.representation == "all"
        else [args.representation]
    )

    # Only inspect the MacBERTh passage tables for the requested
    # representations. Each table may contain many passage vectors.
    tables = []
    for name in db.table_names():
        if not name.startswith("passage_"):
            continue
        if "__macberth__" not in name:
            continue
        if not any(name.startswith(f"passage_{r}__") for r in representations):
            continue
        tables.append(name)

    args.output.mkdir(parents=True, exist_ok=True)

    matches = []
    vectors = []

    for table_name in sorted(tables):
        table = db.open_table(table_name)

        for doc_id in sorted(requested):
            escaped_id = doc_id.replace("'", "''")
            try:
                rows = (
                    table.search()
                    .where(f"doc_id = '{escaped_id}'")
                    .limit(100000)
                    .to_list()
                )
            except Exception as exc:
                print(f"WARNING: {table_name}, {doc_id}: {exc}")
                continue

            for row in rows:
                vector = np.asarray(row["vector"], dtype=np.float32)
                vector_index = len(vectors)
                vectors.append(vector)

                metadata = {
                    key: row.get(key)
                    for key in (
                        "passage_id",
                        "corpus",
                        "doc_id",
                        "pub_year",
                        "word_start",
                        "word_end",
                        "token_start",
                        "token_end",
                        "embedding_model",
                    )
                }
                metadata["table"] = table_name
                metadata["vector_index"] = vector_index
                metadata["vector_dim"] = int(vector.size)
                matches.append(metadata)

                print(
                    f"{doc_id} | {table_name} | "
                    f"year={row.get('pub_year')} | "
                    f"words={row.get('word_start')}:{row.get('word_end')} | "
                    f"dim={vector.size}"
                )

    # Save the vectors as one stacked array and metadata in matching order.
    if vectors:
        dims = {v.size for v in vectors}
        if len(dims) == 1:
            np.save(args.output / "vectors.npy", np.stack(vectors))
        else:
            # Unusual, but preserve vectors if dimensions differ.
            np.savez(
                args.output / "vectors.npz",
                **{f"vector_{i}": v for i, v in enumerate(vectors)},
            )

        with (args.output / "metadata.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(matches, f, ensure_ascii=False, indent=2, default=str)

    found = {m["doc_id"] for m in matches}
    print("\nSummary")
    print(f"Requested document IDs: {len(requested)}")
    print(f"Documents found:        {len(found)}")
    print(f"Passage vectors found:   {len(matches)}")

    for doc_id in sorted(requested - found):
        print(f"NOT FOUND: {doc_id}")

    if matches:
        print(f"Saved output to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
