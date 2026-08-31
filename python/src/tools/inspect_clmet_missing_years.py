#!/usr/bin/env python
"""
inspect_clmet_missing_years.py

Inspect CLMET documents whose pub_year is NULL, derive a usable publication
year from the CLMET source metadata, and update PostgreSQL.

Uses the filepath stored in PostgreSQL, relative to
config.CLMET_CORPUS_INPUT_DIR.

Derivation priority:

    <year>
    <decade>
    <period>

Ranges are converted to their midpoint.

Only documents whose pub_year is currently NULL are updated.
"""

from __future__ import annotations

import re
from pathlib import Path

from lib.corpus_db import get_connection
import lib.corpus_config as config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midpoint(a: int, b: int) -> int:
    """Return integer midpoint, rounded to nearest year."""
    return int(round((a + b) / 2))


def expand_short_year(start: int, end_raw: str) -> int:
    """
    Expand abbreviated CLMET year ranges.

    Examples:

        1730-1  -> 1731
        1740-41 -> 1741
        1760-1  -> 1761
        1773-4  -> 1774
        1780-96 -> 1796
        1820-2  -> 1822
        1810-3  -> 1813
        1904-5  -> 1905
        1888-9  -> 1889
    """

    end = int(end_raw)

    if len(end_raw) == 1:
        # 1730-1 -> 1731
        # 1820-2 -> 1822
        end = (start // 10) * 10 + end

    elif len(end_raw) == 2:
        # 1780-96 -> 1796
        # 1740-41 -> 1741
        end = (start // 100) * 100 + end

    else:
        end = int(end_raw)

    if end < start:
        end += 100

    return end


def clean_metadata_value(value: str) -> str:
    """
    Remove CLMET annotation prefixes while retaining the actual date.

    Examples:

        ?1750      -> 1750
        X1780-96   -> 1780-96
        a1911      -> 1911
    """

    value = value.strip()

    value = (
        value
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    # CLMET uses occasional annotation prefixes such as ?, X, a.
    value = re.sub(r"^[?XaA]+", "", value)

    return value.strip()


def parse_year(value: str | None) -> int | None:
    """
    Parse a CLMET <year> value.

    Examples:

        1750       -> 1750
        ?1750      -> 1750
        1730-1     -> 1731
        1780-96    -> 1788
        1746-71    -> 1759
        1796-1817  -> 1807
    """

    if not value:
        return None

    value = clean_metadata_value(value)

    m = re.fullmatch(r"(\d{4})", value)

    if m:
        return int(m.group(1))

    m = re.fullmatch(
        r"(\d{4})\s*-\s*(\d{1,4})",
        value,
    )

    if m:
        start = int(m.group(1))
        end = expand_short_year(start, m.group(2))
        return midpoint(start, end)

    return None


def parse_decade(value: str | None) -> int | None:
    """
    Parse a CLMET <decade> value.

    Examples:

        1750s -> 1755
        1710s -> 1715
        1911s -> 1916
    """

    if not value:
        return None

    value = clean_metadata_value(value)

    m = re.fullmatch(r"(\d{4})s", value)

    if m:
        return int(m.group(1)) + 5

    return None


def parse_period(value: str | None) -> int | None:
    """
    Parse a CLMET <period> as a last-resort range.

    Examples:

        1710-1780 -> 1745
        1780-1850 -> 1815
        1850-1920 -> 1885
    """

    if not value:
        return None

    value = clean_metadata_value(value)

    m = re.fullmatch(
        r"(\d{4})\s*-\s*(\d{4})",
        value,
    )

    if m:
        return midpoint(
            int(m.group(1)),
            int(m.group(2)),
        )

    return None


# ---------------------------------------------------------------------------
# Metadata reader
# ---------------------------------------------------------------------------

def read_metadata(path: Path) -> dict[str, str]:
    """
    Read CLMET XML-style metadata from the beginning of a text file.
    """

    metadata: dict[str, str] = {}

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            m = re.match(
                r"<(year|decade|period)>(.*?)</\1>",
                line,
                flags=re.IGNORECASE,
            )

            if m:
                metadata[m.group(1).lower()] = (
                    m.group(2).strip()
                )
                continue

            if not line.startswith("<"):
                break

    return metadata


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------

def derive_year(metadata: dict[str, str]) -> tuple[int | None, str | None]:
    """
    Derive a publication year using the agreed priority:

        year -> decade -> period
    """

    year_raw = metadata.get("year")

    if year_raw:
        derived = parse_year(year_raw)

        if derived is not None:
            return derived, "year"

    decade_raw = metadata.get("decade")

    if decade_raw:
        derived = parse_decade(decade_raw)

        if derived is not None:
            return derived, "decade"

    period_raw = metadata.get("period")

    if period_raw:
        derived = parse_period(period_raw)

        if derived is not None:
            return derived, "period"

    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    doc_id,
                    pub_year,
                    title,
                    filepath
                FROM documents
                WHERE corpus = 'clmet'
                  AND pub_year IS NULL
                ORDER BY doc_id
                """
            )

            rows = cur.fetchall()

        root = Path(config.CLMET_CORPUS_INPUT_DIR)

        print("=" * 110)
        print("CLMET DOCUMENTS WITH MISSING pub_year")
        print("=" * 110)
        print()
        print(f"Source root: {root}")
        print(f"Documents:   {len(rows)}")
        print()

        results = []
        missing_files = []

        for doc_id, db_year, title, filepath in rows:

            path = root / filepath

            if not path.exists():
                missing_files.append(
                    (doc_id, title, filepath)
                )
                print(
                    f"MISSING FILE  {doc_id:16} "
                    f"{filepath}"
                )
                continue

            metadata = read_metadata(path)

            derived, source = derive_year(metadata)

            results.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "filepath": filepath,
                    "year_raw": metadata.get("year"),
                    "decade_raw": metadata.get("decade"),
                    "period_raw": metadata.get("period"),
                    "derived": derived,
                    "source": source,
                }
            )

        # -------------------------------------------------------------------
        # Display derivations
        # -------------------------------------------------------------------

        print(
            f"{'DOC ID':18} "
            f"{'YEAR':12} "
            f"{'DECADE':10} "
            f"{'PERIOD':12} "
            f"{'DERIVED':8} "
            f"{'SOURCE':8}"
        )

        print("-" * 110)

        for r in results:
            print(
                f"{r['doc_id']:18} "
                f"{str(r['year_raw'] or '—'):12} "
                f"{str(r['decade_raw'] or '—'):10} "
                f"{str(r['period_raw'] or '—'):12} "
                f"{str(r['derived'] or '???'):8} "
                f"{str(r['source'] or 'NONE'):8}"
            )

        resolved = [
            r
            for r in results
            if r["derived"] is not None
        ]

        unresolved = [
            r
            for r in results
            if r["derived"] is None
        ]

        # -------------------------------------------------------------------
        # Safety checks BEFORE any UPDATE
        # -------------------------------------------------------------------

        print()
        print("=" * 110)
        print("VALIDATION")
        print("=" * 110)
        print()
        print(f"Documents selected: {len(rows):,}")
        print(f"Files found:        {len(results):,}")
        print(f"Resolved:           {len(resolved):,}")
        print(f"Unresolved:         {len(unresolved):,}")
        print(f"Missing files:      {len(missing_files):,}")

        if missing_files:
            print()
            print("ABORTING: source files are missing.")
            conn.rollback()
            return

        if unresolved:
            print()
            print("ABORTING: some documents could not be assigned a year.")

            for r in unresolved:
                print()
                print(f"{r['doc_id']}: {r['title']}")
                print(f"  filepath: {r['filepath']}")
                print(f"  year:     {r['year_raw']}")
                print(f"  decade:   {r['decade_raw']}")
                print(f"  period:   {r['period_raw']}")

            conn.rollback()
            return

        # -------------------------------------------------------------------
        # Update
        # -------------------------------------------------------------------

        print()
        print("=" * 110)
        print("UPDATING DATABASE")
        print("=" * 110)
        print()

        updated = 0

        with conn.cursor() as cur:

            for r in resolved:
                cur.execute(
                    """
                    UPDATE documents
                    SET pub_year = %s
                    WHERE corpus = 'clmet'
                      AND doc_id = %s
                      AND pub_year IS NULL
                    """,
                    (
                        r["derived"],
                        r["doc_id"],
                    ),
                )

                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"Expected exactly one update for "
                        f"{r['doc_id']}, got {cur.rowcount}"
                    )

                updated += 1

                print(
                    f"{r['doc_id']:18} "
                    f"-> {r['derived']} "
                    f"({r['source']})"
                )

        # Commit only after every update succeeded.
        conn.commit()

        print()
        print("=" * 110)
        print("DATABASE UPDATE COMPLETE")
        print("=" * 110)
        print()
        print(f"Rows updated: {updated:,}")

        # -------------------------------------------------------------------
        # Verify
        # -------------------------------------------------------------------

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM documents
                WHERE corpus = 'clmet'
                  AND pub_year IS NULL
                """
            )

            remaining = cur.fetchone()[0]

        print(f"Remaining NULL pub_year: {remaining:,}")

        if remaining != 0:
            print()
            print(
                "WARNING: some CLMET documents still have NULL pub_year."
            )

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


if __name__ == "__main__":
    main()
