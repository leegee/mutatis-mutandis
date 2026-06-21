#!/usr/bin/env python3

import re
import unicodedata

from lib.eebo_db import get_connection
from lib.eebo_logging import logger

logger.info("Enter")

REPLACEMENTS = {
    "aberdene": "Aberdeen",

    "bruxelles": "Brussels",
    "bristoll": "bristol",

    # Cambridge
    "cantabrigiae nov. ang.": "Cambridge",

    "delph": "Delft",

    "edenbrough": "Edinburgh",

    # Hague
    "a la haye": "The Hague",
    "hag├": "The Hague",
    "hague": "The Hague",

    # London
    "london": "London",
    "londini": "London",
    "lond.": "London",
    "lonon": "London",
    "londnon": "London",
    "llundain": "London",
    "a londres": "London",
    "gehenna": "London",

    # Oxford
    "oxon.": "Oxford",

    # Saint Omer
    "s. omers": "Saint-Omer",

    # York
    "yorke": "York",

}


LEADING_NOISE = re.compile(
    r"^[\s\[\(]*"
    r"(?:printed\s+(?:by\s+[^,]+,\s*)?at\s+|at\s+|in\s+)?"
    r"[\s\[\(\.…*]*",
    re.I
)

TRAILING_NOISE = re.compile(
    r"[\s\]\)?\[,;:\.?!\"\'—–\-]+$"
)

DATE_PREFIX = re.compile(
    r"^(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z.]*\s+)?"
    r"\d{1,2}[.,]?\s*\d{4}[.,]?\s*",
    re.I
)

UNCERTAINTY = re.compile(r"\?")

MULTIPLE_PLACE_SEP = re.compile(r"\s+and\s+", re.I)


def clean_place(place: str | None) -> str | None:
    if not place:
        return None

    s = unicodedata.normalize("NFKC", place).strip()

    # Expand editorial corrections [i.e. X] before anything else
    s = re.sub(r"\[i\.e\.\s*([^\]]+)\]", r"\1", s, flags=re.I)
    s = re.sub(r"\[sic\]", "", s, flags=re.I)

    # Drop clearly damaged/partial tokens like "Lon[don pri]nted at"
    # (bracket content that contains lowercase letters mid-word — editorial damage)
    s = re.sub(r"\[[^\]]{1,20}\]", "", s)  # remove short bracketed fragments

    # Strip "re-printed at X" / "printed for X" tails
    s = re.sub(r",?\s*(re-?\s*)?printed\b.*", "", s, flags=re.I)
    s = re.sub(r",?\s*printed\s+for\b.*", "", s, flags=re.I)

    # Strip date prefix  e.g. "March 11. 1643."
    s = DATE_PREFIX.sub("", s)

    # Strip leading noise: brackets, "In ", "At ", "Printed at "
    s = LEADING_NOISE.sub("", s)

    # Strip trailing noise: colons, brackets, commas, punctuation
    s = TRAILING_NOISE.sub("", s)
    s = re.sub(r"[\[\s?i\.?e\.?\s+.+$]", "", s)

    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Null-place markers
    token_check = re.sub(r"[^a-z]", "", s.lower())
    if token_check in ("sl", "sn", "sp"):
        return token_check

    s_clean = UNCERTAINTY.sub("", token_check).strip()

    lower = s_clean.lower()
    lower = TRAILING_NOISE.sub("", lower)
    if lower in REPLACEMENTS:
        return REPLACEMENTS[lower]

    parts = MULTIPLE_PLACE_SEP.split(lower)
    lower = parts[0].strip()

    if lower in REPLACEMENTS:
        return REPLACEMENTS[lower]

    logger.info(f" <<{place}>> ---> <<{lower}>>")

    # Title-case if fully lowercase (artifact of noisy input)
    if s_clean == s_clean.lower():
        s_clean = s_clean.title()

    return s_clean if s_clean else None

def main():
    conn = get_connection(application_name="normalize_places")

    with conn.cursor() as cur:
        cur.execute("""
            DROP TABLE IF EXISTS place_normalization;
            CREATE TABLE place_normalization (
                raw_place text PRIMARY KEY,
                normalized_place text
            )
        """)

        cur.execute("""
            SELECT DISTINCT pub_place
            FROM documents
            WHERE pub_place IS NOT NULL
            ORDER BY pub_place
        """)

        rows = cur.fetchall()

        for (raw_place,) in rows:
            normalized = clean_place(raw_place)
            logger.info(f"{raw_place} \t-->\t\t {normalized}")

            cur.execute("""
                INSERT INTO place_normalization
                    (raw_place, normalized_place)
                VALUES (%s, %s)
                ON CONFLICT (raw_place)
                DO UPDATE
                SET normalized_place = EXCLUDED.normalized_place
            """, (raw_place, normalized))

    conn.commit()

    logger.info(f"Processed {len(rows):,} distinct place strings")

    total = 0
    mapped = 0
    unresolved = []

    conn = get_connection(application_name="normalize_places")

    with conn.cursor() as cur:
        for (raw_place,) in rows:
            total += 1

            normalized = clean_place(raw_place)

            cur.execute("""
                INSERT INTO place_normalization
                    (raw_place, normalized_place)
                VALUES (%s, %s)
                ON CONFLICT (raw_place)
                DO UPDATE
                SET normalized_place = EXCLUDED.normalized_place
            """, (raw_place, normalized))

            if normalized is None:
                unresolved.append(raw_place)
            elif normalized != raw_place:
                mapped += 1

        conn.commit()

    # print()
    # print("=== Place normalization summary ===")
    # print(f"Distinct places:     {total:,}")
    # print(f"Mapped/normalized:   {mapped:,}")
    # print(f"Unresolved:          {len(unresolved):,}")
    # print()

    # if unresolved:
    #     print("Top unresolved values:")
    #     for place in sorted(unresolved)[:100]:
    #         print(f"  {place}")

    # select distinct(normalized_place) from place_normalization;

if __name__ == "__main__":
    main()
