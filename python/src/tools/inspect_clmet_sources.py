#!/usr/bin/env python
"""
inspect_clmet_sources.py

Compare the CLMET source files referenced by PostgreSQL with the
CLMET documents currently registered in the database.

The database's filepath field is authoritative: no filename guessing
or recursive source-directory search is performed.
"""

from pathlib import Path
import re

from lib.corpus_db import get_connection
import  lib.corpus_config as config


def approx_tokens(text: str) -> int:
    """Deliberately simple sizing estimate, not corpus tokenisation."""
    return len(re.findall(r"\S+", text))


def main():
    conn = get_connection()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                doc_id,
                pub_year,
                title,
                token_count,
                filepath
            FROM documents
            WHERE corpus = 'clmet'
            ORDER BY doc_id
        """)
        rows = cur.fetchall()

    conn.close()

    print(f"CLMET documents in DB: {len(rows)}")
    print()

    total_bytes = 0
    total_chars = 0
    total_words = 0
    found = 0

    for (
        doc_id,
        year,
        title,
        db_tokens,
        filepath,
    ) in rows:

        if not filepath:
            print(
                f"MISSING-PATH  {doc_id:12} "
                f"DB tokens={db_tokens or 0:8,}  "
                f"{title}"
            )
            continue

        path = Path(config.CLMET_CORPUS_INPUT_DIR) / filepath

        if not path.exists():
            print(
                f"MISSING       {doc_id:12} "
                f"DB tokens={db_tokens or 0:8,}  "
                f"{path}"
            )
            continue

        if not path.is_file():
            print(
                f"NOT-A-FILE    {doc_id:12} "
                f"{path}"
            )
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            print(
                f"ERROR         {doc_id:12} "
                f"{path}: {exc}"
            )
            continue

        size = path.stat().st_size
        chars = len(text)
        words = approx_tokens(text)

        found += 1
        total_bytes += size
        total_chars += chars
        total_words += words

        ratio = (
            db_tokens / words
            if words
            else 0
        )

        print(
            f"{doc_id:12} "
            f"{year or '----':4} "
            f"{size / 1024:9.1f} KB "
            f"{chars:10,} chars "
            f"{words:10,} words "
            f"DB={db_tokens or 0:8,} "
            f"ratio={ratio:5.2f}  "
            f"{path.name}"
        )

    print()
    print("SUMMARY")
    print(f"  DB documents:     {len(rows):,}")
    print(f"  Files found:      {found:,}")
    print(f"  Files missing:    {len(rows) - found:,}")
    print(f"  Source size:      {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Characters:       {total_chars:,}")
    print(f"  Approx. words:    {total_words:,}")


if __name__ == "__main__":
    main()
