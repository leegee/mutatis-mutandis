#!/usr/bin/env python
"""
build_spelling_map_from_fastText.py

Stage 2 (refined): Generate the canonical spelling_map table from the global fastText model
for a curated list of keywords, with Levenshtein filtering and an explicit blacklist.

- Loads global fastText skipgram model (Stage 1)
- Queries nearest neighbours for keywords only
- Determines canonical form using filtered neighbours
- Inserts results into `spelling_map` table
- Updates `tokens.canonical` column
- Prints a full report at the end
- Supports dry-run mode (--dry)
"""

from pathlib import Path
import argparse
from tqdm import tqdm
import fasttext
import Levenshtein

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger


# Explicit blacklist of words to ignore as orthographic variants
BLACKLIST = {
    "ureasonable",
    "indiffeasible",
    "afreedom",
    "divineinfluence",
    "supremacy",  # example additional semantic mismatches
    # Add more problematic tokens here
}


def load_fasttext_model(model_path: Path) -> fasttext.FastText:
    logger.info(f"Loading global fastText model from {model_path}")
    return fasttext.load_model(str(model_path))


def determine_canonical(neighbours: list[tuple[float, str]]) -> str | None:
    """Pick canonical token from a set of neighbours: first neighbour if exists."""
    return neighbours[0][1] if neighbours else None


def insert_spelling_map(conn, variant: str, canonical: str, dry: bool = False) -> None:
    """Insert one variant → canonical mapping into spelling_map table."""
    if dry:
        logger.debug(f"[DRY] Insert spelling_map: {variant} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO spelling_map(variant, canonical, concept_type)
            VALUES (%s, %s, 'orthographic')
            ON CONFLICT (variant) DO UPDATE SET canonical = EXCLUDED.canonical;
            """,
            (variant, canonical),
        )


def update_tokens_canonical(conn, token: str, canonical: str, dry: bool = False) -> None:
    """Update tokens.canonical column for a single token."""
    if dry:
        logger.debug(f"[DRY] Update tokens.canonical: {token} -> {canonical}")
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE tokens
            SET canonical = %s
            WHERE token = %s;
            """,
            (canonical, token),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build canonical spelling_map from global fastText model for curated keywords"
    )
    parser.add_argument("--model_path", type=str, default=str(config.FASTTEXT_GLOBAL_MODEL_PATH),
                        help=f"Path to global fastText model (default: {config.FASTTEXT_GLOBAL_MODEL_PATH})")
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
                        help=f"Number of nearest neighbours to consider (default: {config.TOP_K})")
    parser.add_argument("--dry", action="store_true", help="Dry run: do not commit to DB")
    parser.add_argument("--max_lev_distance", type=int, default=3,
                        help="Maximum Levenshtein distance for orthographic neighbours")
    args = parser.parse_args()

    if args.dry:
        logger.info("DRY RUN: no database writes will occur")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    model = load_fasttext_model(model_path)

    # Use only curated keywords from config
    keywords = getattr(config, "KEYWORDS_TO_NORMALISE", [])
    if not keywords:
        logger.error("No keywords defined in config.KEYWORDS_TO_NORMALISE")
        return

    report = []

    # Open a DB connection
    with eebo_db.get_connection(application_name="spelling_map_builder") as conn:
        logger.info(f"Processing {len(keywords)} keywords for canonicalisation")

        for keyword in tqdm(keywords, desc="Keywords processed"):
            try:
                neighbours = model.get_nearest_neighbors(keyword, k=args.top_k)
            except KeyError:
                logger.warning(f"Keyword '{keyword}' not found in fastText vocabulary")
                continue

            # Filter neighbours by Levenshtein distance and blacklist
            filtered: list[str] = [
                n for _sim, n in neighbours
                if Levenshtein.distance(keyword.lower(), n.lower()) <= args.max_lev_distance
                and n.lower() not in BLACKLIST
            ]

            if not filtered:
                logger.warning(f"No valid orthographic neighbours for '{keyword}', skipping")
                continue

            canonical = filtered[0]  # pick first filtered as canonical

            insert_spelling_map(conn, keyword, canonical, dry=args.dry)
            update_tokens_canonical(conn, keyword, canonical, dry=args.dry)

            report.append({
                "keyword": keyword,
                "canonical": canonical,
                "filtered_neighbours": filtered,
                "original_neighbours": [n for _, n in neighbours],
            })

            logger.info(f"Keyword: {keyword}")
            logger.info(f"  Canonical: {canonical}")
            logger.info(f"  Filtered neighbours: {', '.join(filtered)}")
            logger.info(f"  Original neighbours: {', '.join([n for _, n in neighbours])}")
            logger.info("-" * 60)

        if not args.dry:
            conn.commit()
            logger.info("Database updated with canonical spellings")

    logger.info("Canonicalisation complete")


if __name__ == "__main__":
    main()
