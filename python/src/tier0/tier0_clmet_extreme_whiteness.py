#!/usr/bin/env python
"""
tier0_clmet_extreme_whiteness.py

Create extracted CLMET documents containing larger contexts around
bodily-whiteness concordance hits and insert them into PostgreSQL.

This is deliberately independent of the existing EEBO/ECCO ingestion
pipeline.

For each CLMET source document:

    source text
        |
        +-- hit 1 --> large context
        +-- hit 2 --> large context
        +-- hit 3 --> large context
                         |
                         v
             context <SEP> context <SEP> context
                         |
                         v
                    documents

The resulting document receives:

    doc_id = "CLMET3" + original document id

and retains the original source filepath.

No tokens are inserted.

The extracted token count is calculated for logging only; it is not
stored separately in the database.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from psycopg import sql

import lib.corpus_config as config
import lib.corpus_db as corpus_db
from lib.corpus_logging import logger


CORPUS_NAME = "clmet"

CLMET_TEXT_DIR = config.CLMET_CORPUS_INPUT_DIR / "txt"

CONCORDANCE_FILE = (
    config.CLMET_CORPUS_INPUT_DIR
    / "extreme_whiteness"
    / "bodily_whiteness_concordance.csv"
)

# Number of tokens on each side of each occurrence.
CONTEXT_TOKENS = 2000

SEPARATOR = "<SEP>"

# Simple tokenizer for locating occurrences and counting extracted tokens.
# The original source text is retained when constructing contexts.
WORD_RE = re.compile(r"[A-Za-zÀ-ÿ'-]+")


# ---------------------------------------------------------------------------
# CSV field handling
# ---------------------------------------------------------------------------

DOC_ID_FIELDS = (
    "doc_id",
    "document_id",
    "document",
    "doc",
    "filename",
    "file",
    "filepath",
    "path",
)

HIT_FIELDS = (
    "whiteness_surface",
    "whiteness_lemma",
    "hit",
    "keyword",
    "target",
    "term",
    "match",
    "matched",
)


def resolve_column(
    fieldnames: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    """Resolve a CSV column name case-insensitively."""

    lowered = {
        field.strip().lower(): field
        for field in fieldnames
        if field
    }

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


# ---------------------------------------------------------------------------
# Document/path resolution
# ---------------------------------------------------------------------------

def normalise_doc_id(value: str) -> str:
    """
    Turn a CSV document reference into the source document identifier.

    For example:

        CLMET3_1_1_42.txt

    becomes:

        CLMET3_1_1_42
    """

    value = value.strip()
    value = value.replace("\\", "/")

    value = Path(value).name

    if value.lower().endswith(".txt"):
        value = value[:-4]

    return value


def find_source_file(doc_ref: str) -> Path | None:
    """
    Locate the CLMET source text.

    Try the supplied value as a relative path first, then as a filename
    beneath txt/, and finally search recursively beneath txt/.
    """

    ref = doc_ref.replace("\\", "/")

    candidates: list[Path] = []

    p = Path(ref)

    if not p.is_absolute():
        candidates.append(config.CLMET_CORPUS_INPUT_DIR / p)
        candidates.append(CLMET_TEXT_DIR / p)

    name = Path(ref).name

    if not name.lower().endswith(".txt"):
        name += ".txt"

    candidates.append(CLMET_TEXT_DIR / name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    matches = list(CLMET_TEXT_DIR.rglob(name))

    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def token_spans(text: str) -> list[tuple[int, int, str]]:
    """
    Return word spans in the original source text.

    Character offsets allow us to extract the original text while preserving
    punctuation and whitespace.
    """

    return [
        (m.start(), m.end(), m.group(0))
        for m in WORD_RE.finditer(text)
    ]


def locate_surface_occurrences(
    text: str,
    surface: str,
) -> list[int]:
    """
    Locate exact lexical occurrences of a concordance surface form.

    Matching is case-insensitive and word-boundary based.

    Returns character offsets.
    """

    if not surface:
        return []

    escaped = re.escape(surface.strip())

    pattern = re.compile(
        rf"(?<![A-Za-zÀ-ÿ'-])"
        rf"{escaped}"
        rf"(?![A-Za-zÀ-ÿ'-])",
        re.IGNORECASE,
    )

    return [
        match.start()
        for match in pattern.finditer(text)
    ]


def context_range_for_position(
    spans: list[tuple[int, int, str]],
    char_pos: int,
    context_tokens: int,
) -> tuple[int, int] | None:
    """
    Return the character range for a context centred on char_pos.
    """

    if not spans:
        return None

    target_idx = None

    for i, (start, end, _) in enumerate(spans):
        if start <= char_pos < end:
            target_idx = i
            break

        if start > char_pos:
            target_idx = i
            break

    if target_idx is None:
        return None

    start_idx = max(
        0,
        target_idx - context_tokens,
    )

    end_idx = min(
        len(spans),
        target_idx + context_tokens + 1,
    )

    return (
        spans[start_idx][0],
        spans[end_idx - 1][1],
    )


def merge_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Merge overlapping or touching source ranges.
    """

    if not ranges:
        return []

    ranges = sorted(ranges)

    merged: list[tuple[int, int]] = []

    start, end = ranges[0]

    for next_start, next_end in ranges[1:]:

        if next_start <= end:
            end = max(end, next_end)
            continue

        merged.append((start, end))

        start = next_start
        end = next_end

    merged.append((start, end))

    return merged


def extract_document_contexts(
    text: str,
    occurrences: list[dict[str, str]],
    context_tokens: int,
) -> tuple[str, int, int]:
    """
    Extract large contexts around concordance occurrences.

    For each concordance occurrence, locate the corresponding surface form
    in the source and extract context_tokens on either side.

    Returns:

        extracted_text
        extracted_token_count
        occurrence_count
    """

    spans = token_spans(text)

    if not spans:
        return "", 0, 0

    ranges: list[tuple[int, int]] = []
    occurrence_count = 0

    # Track which source occurrence of each surface form has already been
    # consumed. This means repeated concordance rows for "white" map onto
    # successive occurrences in the source rather than all selecting the
    # first "white".
    consumed_positions: dict[str, int] = defaultdict(int)

    for occurrence in occurrences:

        surface = occurrence.get("whiteness_surface", "").strip()

        if not surface:
            continue

        positions = locate_surface_occurrences(
            text,
            surface,
        )

        position_index = consumed_positions[surface]

        if position_index >= len(positions):
            logger.warning(
                "[clmet] Could not locate concordance occurrence "
                f"#{position_index + 1} of '{surface}' in source text"
            )
            continue

        pos = positions[position_index]
        consumed_positions[surface] += 1

        context_range = context_range_for_position(
            spans,
            pos,
            context_tokens,
        )

        if context_range is None:
            continue

        ranges.append(context_range)
        occurrence_count += 1

    if not ranges:
        return "", 0, 0

    merged_ranges = merge_ranges(ranges)

    extracted_parts = [
        text[start:end].strip()
        for start, end in merged_ranges
    ]

    extracted_text = f" {SEPARATOR} ".join(
        extracted_parts
    )

    extracted_token_count = len(
        WORD_RE.findall(extracted_text)
    )

    return (
        extracted_text,
        extracted_token_count,
        occurrence_count,
    )



# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def metadata_from_text_file(
    doc_id: str,
    filepath: Path,
) -> dict:
    """
    Construct metadata for the derived CLMET document.

    The first pass deliberately does not attempt to build a separate
    CLMET metadata parser.
    """

    return {
        "corpus": CORPUS_NAME,
        "doc_id": f"CLMET3{doc_id}",
        "filepath": filepath.relative_to(
            config.CLMET_CORPUS_INPUT_DIR
        ).as_posix(),
        "title": None,
        "author": None,
        "pub_year": None,
        "publisher": None,
        "pub_place": None,
        "source_date_raw": None,
        "token_count": None,
        "lang": "eng",
    }


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def insert_documents(rows: list[dict]) -> None:
    """Insert extracted documents into PostgreSQL."""

    if not rows:
        return

    columns = [
        "corpus",
        "doc_id",
        "filepath",
        "title",
        "author",
        "pub_year",
        "publisher",
        "pub_place",
        "source_date_raw",
        "token_count",
        "lang",
        "extracted_text",
    ]

    stmt = sql.SQL(
        """
        INSERT INTO documents ({fields})
        VALUES ({values})
        ON CONFLICT (doc_id) DO UPDATE SET
            corpus = EXCLUDED.corpus,
            filepath = EXCLUDED.filepath,
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            pub_year = EXCLUDED.pub_year,
            publisher = EXCLUDED.publisher,
            pub_place = EXCLUDED.pub_place,
            source_date_raw = EXCLUDED.source_date_raw,
            token_count = EXCLUDED.token_count,
            lang = EXCLUDED.lang,
            extracted_text = EXCLUDED.extracted_text
        """
    ).format(
        fields=sql.SQL(", ").join(
            sql.Identifier(column)
            for column in columns
        ),
        values=sql.SQL(", ").join(
            sql.Placeholder(column)
            for column in columns
        ),
    )

    with corpus_db.get_connection(
        application_name="tier0-clmet-extreme-whiteness",
    ) as conn:

        with conn.transaction():

            with conn.cursor() as cur:

                for row in rows:
                    cur.execute(stmt, row)


def log_dry_run(rows: list[dict]) -> None:
    """
    Log the rows that would have been inserted.

    This intentionally logs the complete extracted text so that --dry-run
    can be used to inspect the actual PostgreSQL payload before writing it.
    """

    logger.info(
        f"[clmet] DRY RUN: {len(rows):,} documents would be inserted"
    )

    for row in rows:

        logger.debug(
            "[clmet] DRY RUN INSERT documents: "
            f"{row}"
        )


def load_concordance(
    path: Path,
) -> dict[str, list[dict[str, str]]]:
    """
    Load the concordance and group occurrence rows by source document.

    The concordance contains both a document identifier and a source-file
    column. The source-file value is retained because it is the authoritative
    way to locate the corresponding CLMET text.

    We deliberately retain one row per concordance occurrence. Do not
    deduplicate lexical terms here: two rows containing "white" represent
    two separate occurrences.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"Concordance file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(
                f"No CSV header found in {path}"
            )

        doc_column = resolve_column(
            reader.fieldnames,
            DOC_ID_FIELDS,
        )

        file_column = resolve_column(
            reader.fieldnames,
            ("file", "filepath", "path", "filename"),
        )

        hit_column = resolve_column(
            reader.fieldnames,
            HIT_FIELDS,
        )

        if doc_column is None:
            raise RuntimeError(
                "Could not identify the document column in "
                f"{path}.\n\nAvailable columns:\n"
                + "\n".join(
                    f"  {x}"
                    for x in reader.fieldnames
                )
            )

        if file_column is None:
            raise RuntimeError(
                "Could not identify the source file column in "
                f"{path}.\n\nAvailable columns:\n"
                + "\n".join(
                    f"  {x}"
                    for x in reader.fieldnames
                )
            )

        if hit_column is None:
            raise RuntimeError(
                "Could not identify the hit column in "
                f"{path}.\n\nAvailable columns:\n"
                + "\n".join(
                    f"  {x}"
                    for x in reader.fieldnames
                )
            )

        logger.info(
            f"[clmet] concordance document column: {doc_column}"
        )

        logger.info(
            f"[clmet] concordance file column: {file_column}"
        )

        logger.info(
            f"[clmet] concordance hit column: {hit_column}"
        )

        grouped: dict[str, list[dict[str, str]]] = (
            defaultdict(list)
        )

        for row in reader:

            doc_value = (row.get(doc_column) or "").strip()
            source_file = (row.get(file_column) or "").strip()
            hit_value = (row.get(hit_column) or "").strip()

            if not doc_value:
                continue

            doc_id = normalise_doc_id(doc_value)

            if not doc_id:
                continue

            if not source_file:
                logger.warning(
                    f"[clmet] No source file recorded for "
                    f"document {doc_id}"
                )
                continue

            if not hit_value:
                continue

            occurrence = {
                key: (value or "").strip()
                for key, value in row.items()
                if key
            }

            # Canonical fields used by the extraction stage.
            occurrence["source_file"] = source_file
            occurrence["whiteness_surface"] = hit_value

            grouped[doc_id].append(occurrence)

    return dict(grouped)


def process(
    concordance_path: Path,
    limit: int | None = None,
    dry_run: bool = False,
    context_tokens: int = CONTEXT_TOKENS,
) -> None:

    grouped = load_concordance(concordance_path)

    logger.info(
        f"[clmet] Documents represented in concordance: "
        f"{len(grouped):,}"
    )

    rows: list[dict] = []

    processed = 0
    missing = 0
    no_hits = 0
    total_extracted_tokens = 0
    total_occurrences = 0

    for doc_id, occurrences in grouped.items():

        if limit is not None and processed >= limit:
            break

        source_file = occurrences[0].get("source_file", "").strip()

        if source_file is None:
            logger.warning( f"[clmet] Source text not found for {doc_id}" )
            missing += 1
            continue

        source_path = find_source_file(source_file)

        if source_path is None:
            logger.warning(
                f"[clmet] Source text not found for {doc_id}: "
                f"{source_file}"
            )
            missing += 1
            continue

        try:
            text = source_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:
            logger.warning(
                f"[clmet] Failed reading {source_path}: {exc}"
            )
            continue

        (
            extracted_text,
            extracted_count,
            occurrence_count,
        ) = extract_document_contexts(
            text,
            occurrences,
            context_tokens,
        )

        if not extracted_text:
            no_hits += 1

            logger.warning(
                f"[clmet] No source occurrences found for {doc_id}"
            )

            continue

        metadata = metadata_from_text_file(
            doc_id,
            source_path,
        )

        metadata["extracted_text"] = extracted_text

        # For this first pass, token_count is the size of the derived
        # extracted document. We are not adding a separate
        # extracted_token_count column.
        metadata["token_count"] = extracted_count

        rows.append(metadata)

        processed += 1
        total_extracted_tokens += extracted_count
        total_occurrences += occurrence_count

        logger.info(
            f"[clmet] {doc_id}: "
            f"{len(occurrences)} concordance occurrences, "
            f"{occurrence_count} located, "
            f"{extracted_count:,} extracted tokens"
        )

        if dry_run:
            logger.debug(
                "[clmet] DRY RUN document:\n"
                f"  doc_id: {metadata['doc_id']}\n"
                f"  filepath: {metadata['filepath']}\n"
                f"  token_count: {metadata['token_count']}\n"
                f"  extracted_text:\n{extracted_text}"
            )

    logger.info(
        f"[clmet] Prepared {len(rows):,} extracted documents"
    )

    if dry_run:
        logger.info(
            f"[clmet] DRY RUN: {len(rows):,} documents "
            "would be inserted"
        )
    else:
        insert_documents(rows)

    print()
    print("=" * 72)

    if dry_run:
        print("CLMET EXTREME-WHITENESS EXTRACTION DRY RUN")
    else:
        print("CLMET EXTREME-WHITENESS EXTRACTION COMPLETE")

    print("=" * 72)
    print()

    print(f"Concordance documents:     {len(grouped):,}")
    print(f"Documents processed:       {processed:,}")
    print(f"Missing source files:      {missing:,}")
    print(f"No source hits found:      {no_hits:,}")
    print(f"Extracted documents:       {len(rows):,}")
    print(f"Concordance occurrences:   {total_occurrences:,}")
    print(f"Extracted tokens:          {total_extracted_tokens:,}")
    print(
        f"Context size:              "
        f"{context_tokens:,} tokens each side"
    )
    print(f"Separator:                 {SEPARATOR}")
    print(f"Dry run:                   {dry_run}")
    print()
    print(f"Concordance:               {concordance_path}")
    print(f"CLMET root:                {config.CLMET_CORPUS_INPUT_DIR}")
    print()



def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Extract large contexts around CLMET extreme-whiteness "
            "concordance occurrences and insert them into PostgreSQL."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of source documents to process.",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=CONTEXT_TOKENS,
        help=(
            "Number of tokens on each side of each occurrence "
            f"(default: {CONTEXT_TOKENS})."
        ),
    )

    parser.add_argument(
        "--concordance",
        type=Path,
        default=CONCORDANCE_FILE,
        help="Path to bodily_whiteness_concordance.csv.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Do not write to PostgreSQL. Log the documents that "
            "would have been inserted."
        ),
    )

    args = parser.parse_args()

    process(
        concordance_path=args.concordance,
        limit=args.limit,
        dry_run=args.dry_run,
        context_tokens=args.context,
    )


if __name__ == "__main__":
    main()

