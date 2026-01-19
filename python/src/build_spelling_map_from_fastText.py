#!/usr/bin/env python
"""
build_spelling_map_from_fastText.py

Stage 2 (refined): Generate the canonical spelling_map table from the global fastText model
for a curated list of keywords, and update tokens.canonical.

- Loads global fastText skipgram model (Stage 1)
- Queries nearest neighbours for keywords only
- Determines canonical form
- Inserts results into `spelling_map` table
- Updates `tokens.canonical` column
- Prints a full report at the end
- Supports dry-run mode (--dry)
"""

from pathlib import Path
import argparse
from tqdm import tqdm
import fasttext
import numpy as np

from lib import eebo_db, eebo_config as config
from lib.eebo_logging import logger


def load_fasttext_model(model_path: Path):
    logger.info(f"Loading global fastText model from {model_path}")
    return fasttext.load_model(str(model_path))


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def determine_canonical(neighbours):
    """Pick canonical token from a set of variants: highest similarity (first neighbour)."""
    return neighbours[0][1]


def insert_spelling_map(conn, variant, canonical, dry: bool = False):
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


def update_tokens_canonical(conn, token, canonical, dry: bool = False):
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


def main():
    parser = argparse.ArgumentParser(
        description="Build canonical spelling_map from global fastText model for keywords only"
    )
    parser.add_argument("--model_path", type=str, default=str(config.FASTTEXT_GLOBAL_MODEL_PATH),
                        help=f"Path to global fastText model (default: {config.FASTTEXT_GLOBAL_MODEL_PATH})")
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
                        help=f"Number of nearest neighbours to consider (default: {config.TOP_K})")
    parser.add_argument("--dry", action="store_true", help="Dry run: do not commit to DB")
    args = parser.parse_args()

    if args.dry:
        logger.info("DRY RUN: no database writes will occur")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    model = load_fasttext_model(model_path)

    # We'll only map keywords defined in config
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

            canonical = determine_canonical(neighbours)

            insert_spelling_map(conn, keyword, canonical, dry=args.dry)
            update_tokens_canonical(conn, keyword, canonical, dry=args.dry)

            # Prepare report entry
            report.append({
                "keyword": keyword,
                "canonical": canonical,
                "neighbours": [n for _, n in neighbours]
            })

        if not args.dry:
            conn.commit()
            logger.info("Database updated with canonical spellings")

    # Print full report
    print("\nCanonicalisation Report:\n")
    for entry in report:
        print(f"Keyword: {entry['keyword']}")
        print(f"  Canonical: {entry['canonical']}")
        print(f"  Neighbours considered: {', '.join(entry['neighbours'])}")
        print("-" * 60)

    logger.info("Canonicalisation complete")


if __name__ == "__main__":
    main()
