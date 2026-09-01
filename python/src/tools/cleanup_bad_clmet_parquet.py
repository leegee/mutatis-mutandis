"""
cleanup_bad_clmet_parquet.py

Remove observations belonging to the CLMET corpus from the Tier 1
Parquet observation stores.

The root is taken from lib.corpus_config.EVENTSTORE_T1_PATH.

Expected layout:

    out/events/
        tier1_shard0/
            year=1476/
                part-....parquet
            year=1477/
                ...
        tier1_shard1/
            ...

The script removes rows where corpus == "clmet" and rewrites only
Parquet parts which actually contain CLMET observations.

Default mode is DRY RUN.

Usage
-----

    python cleanup_bad_clmet_parquet.py

    python cleanup_bad_clmet_parquet.py --apply

    python cleanup_bad_clmet_parquet.py --apply --backup

"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lib.corpus_config import EVENTSTORE_T1_PATH
from lib.parquet_observation_backend import write_observation_parquet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(EVENTSTORE_T1_PATH)

BAD_CORPUS = "clmet"

# Safety: only inspect Parquet files below this configured root.
PARQUET_PATTERN = "*.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_parquet_files(root: Path) -> list[Path]:
    """Return every Parquet part beneath the configured event-store root."""
    if not root.exists():
        raise FileNotFoundError(
            f"Configured event-store root does not exist: {root}"
        )

    return sorted(root.rglob(PARQUET_PATTERN))


def count_corpus_rows(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> tuple[int, int]:
    """
    Return (total_rows, bad_rows) for one Parquet file.
    """
    result = con.execute(
        """
        SELECT
            count(*) AS total_rows,
            count(*) FILTER (WHERE lower(corpus) = ?) AS bad_rows
        FROM read_parquet(?)
        """,
        [BAD_CORPUS, str(path)],
    ).fetchone()

    return int(result[0]), int(result[1])


def remove_bad_rows(
    table: pa.Table,
) -> pa.Table:
    """Return table with all CLMET observations removed."""
    corpus = pc.utf8_lower(table["corpus"])
    mask = pc.not_equal(corpus, BAD_CORPUS)
    return table.filter(mask)


def describe_file(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def scan(root: Path) -> tuple[list[Path], int, int, int]:
    """
    Scan the entire event store.

    Returns:

        affected_files
        total_rows
        bad_rows
        surviving_rows
    """
    files = find_parquet_files(root)

    if not files:
        print(f"No Parquet files found below:\n  {root}")
        return [], 0, 0, 0

    con = duckdb.connect(database=":memory:")

    affected: list[Path] = []
    total_rows = 0
    bad_rows = 0

    try:
        for path in files:
            total, bad = count_corpus_rows(con, path)

            total_rows += total
            bad_rows += bad

            if bad:
                affected.append(path)
                print(
                    f"  {describe_file(path, root)}"
                    f"  total={total:,}"
                    f"  clmet={bad:,}"
                    f"  keep={total - bad:,}"
                )
    finally:
        con.close()

    surviving_rows = total_rows - bad_rows

    return affected, total_rows, bad_rows, surviving_rows


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


def rewrite_file(
    path: Path,
    *,
    backup: bool,
) -> tuple[int, int]:
    """
    Rewrite one Parquet file without CLMET rows.

    Returns:

        (old_row_count, new_row_count)
    """
    table = pq.read_table(path)

    old_count = table.num_rows

    cleaned = remove_bad_rows(table)

    new_count = cleaned.num_rows

    if new_count == old_count:
        raise RuntimeError(
            f"rewrite_file called for unaffected file: {path}"
        )

    # Optional backup sits beside the original.
    backup_path = path.with_suffix(path.suffix + ".before_clmet_cleanup")

    if backup:
        if backup_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing backup:\n  {backup_path}"
            )
        shutil.copy2(path, backup_path)

    # Write to a temporary file in the same directory so replacement is
    # on the same filesystem.
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".parquet.tmp",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_observation_parquet(
            cleaned,
            tmp_path,
        )

        # Atomic replacement on the same filesystem.
        tmp_path.replace(path)

    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise

    return old_count, new_count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Remove CLMET observations from the configured Tier 1 "
            "Parquet event store."
        )
    )

    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite affected Parquet files.",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
        help=(
            "When used with --apply, retain a .before_clmet_cleanup "
            "copy of every rewritten Parquet file."
        ),
    )

    args = parser.parse_args()

    root = ROOT.resolve()

    print()
    print("CLMET Parquet cleanup")
    print("=====================")
    print()
    print(f"Configured root:")
    print(f"  {root}")
    print()
    print(f"Removing corpus:")
    print(f"  {BAD_CORPUS!r}")
    print()

    if args.apply:
        print("MODE: APPLY")
    else:
        print("MODE: DRY RUN")
    print()

    affected, total_rows, bad_rows, surviving_rows = scan(root)

    print()
    print("----------------------------------------")
    print("Scan summary")
    print("----------------------------------------")
    print(f"Total Parquet rows:       {total_rows:,}")
    print(f"CLMET rows to remove:     {bad_rows:,}")
    print(f"Rows remaining:           {surviving_rows:,}")
    print(f"Affected Parquet files:   {len(affected):,}")
    print()

    if not affected:
        print("No CLMET observations found. Nothing to do.")
        return 0

    if not args.apply:
        print("DRY RUN ONLY — no files were changed.")
        print()
        print("To perform the cleanup:")
        print("  python cleanup_bad_clmet_parquet.py --apply")
        if not args.backup:
            print()
            print(
                "For an additional copy of every rewritten part:"
            )
            print(
                "  python cleanup_bad_clmet_parquet.py --apply --backup"
            )
        return 0

    print("Rewriting affected files...")
    print()

    removed = 0
    old_total = 0
    new_total = 0

    for path in affected:
        old_count, new_count = rewrite_file(
            path,
            backup=args.backup,
        )

        removed_here = old_count - new_count

        old_total += old_count
        new_total += new_count
        removed += removed_here

        print(
            f"  cleaned {describe_file(path, root)}"
            f"  removed={removed_here:,}"
        )

    print()
    print("----------------------------------------")
    print("Cleanup summary")
    print("----------------------------------------")
    print(f"Files rewritten:          {len(affected):,}")
    print(f"Rows before:              {old_total:,}")
    print(f"Rows removed:             {removed:,}")
    print(f"Rows remaining:           {new_total:,}")
    print()

    if removed != bad_rows:
        raise RuntimeError(
            "SAFETY CHECK FAILED: number of rows removed during rewrite "
            f"({removed:,}) does not match scan ({bad_rows:,})."
        )

    print(
        f"Safety check passed: exactly {removed:,} CLMET rows removed."
    )

    if args.backup:
        print()
        print(
            "Backups were retained beside rewritten files with suffix:"
        )
        print("  .before_clmet_cleanup")

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
