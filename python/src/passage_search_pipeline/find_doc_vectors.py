
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import lancedb
import numpy as np

from lib.corpus_config import LANCE_INDEXES_DIR, OUT_DIR
from lib.corpus_db import get_connection
from passage_search_pipeline import passages_search as ps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find existing LanceDB passage vectors by document ID."
    )
    parser.add_argument( "doc_ids", nargs="+", help="Document IDs to find, e.g. CLMET3255 A12345", )
    parser.add_argument( "--representation", choices=["l8", "last", "mean4", "all"], default="all", )
    parser.add_argument( "--output", type=Path, default=OUT_DIR / "doc_vectors" )
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
    tables = [
        name
        for name in db.list_tables().tables
        if name.startswith("passage_")
        and "__macberth__" in name
        and any(
            name.startswith(f"passage_{representation}__")
            for representation in representations
        )
    ]

    print(f"Index: {LANCE_INDEXES_DIR}")
    print(f"Matching MacBERTh tables: {len(tables)}")

    args.output.mkdir(parents=True, exist_ok=True)

    matches = []
    vectors = []

    with get_connection(application_name="find-doc-vectors") as conn:
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

                    # Preserve the original vector metadata.
                    metadata_row = {
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
                    metadata_row["table"] = table_name
                    metadata_row["vector_index"] = vector_index
                    metadata_row["vector_dim"] = int(vector.size)

                    # Retrieve the associated passage text and bibliographic
                    # metadata from the source corpus.
                    try:
                        source = ps.fetch_passage_text(conn, row)
                    except Exception as exc:
                        print(
                            f"WARNING: Could not fetch passage text for "
                            f"{row.get('passage_id')}: {exc}"
                        )
                        source = {}

                    metadata_row["title"] = source.get("title")
                    metadata_row["author"] = source.get("author")
                    metadata_row["filepath"] = source.get("filepath")
                    metadata_row["passage_text"] = source.get("passage_text")

                    matches.append(metadata_row)

                    print(
                        f"\n{doc_id} | {table_name} | "
                        f"year={row.get('pub_year')} | "
                        f"words={row.get('word_start')}:{row.get('word_end')} | "
                        f"dim={vector.size}"
                    )
                    print(f"  passage_id={row.get('passage_id')}")
                    print(
                        f"  title={metadata_row['title'] or '(untitled)'}"
                    )
                    print(
                        f"  author={metadata_row['author'] or '(unknown)'}"
                    )
                    print(
                        f"  filepath={metadata_row['filepath'] or '(unknown)'}"
                    )
                    print(
                        f"  tokens={row.get('token_start')}:"
                        f"{row.get('token_end')}"
                    )
                    print(
                        f"  {metadata_row['passage_text'] or '[Text unavailable]'}"
                    )

    # Save vectors in the same order as the metadata records.
    if vectors:
        dims = {v.size for v in vectors}
        if len(dims) == 1:
            np.save(args.output / "vectors.npy", np.stack(vectors))
        else:
            # Preserve vectors if dimensions differ.
            np.savez(
                args.output / "vectors.npz",
                **{f"vector_{i}": v for i, v in enumerate(vectors)},
            )

        with (args.output / "metadata.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                matches,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

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
