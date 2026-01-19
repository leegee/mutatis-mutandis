#!/usr/bin/env python
"""
build_spelling_map_from_fastText.py

Stage 2: Generate spelling_map entries using a FIXED set of canonical heads.

- Canonical heads come ONLY from config.KEYWORDS_TO_NORMALISE
- fastText is used ONLY to discover orthographic variants
- Canonical choice is NEVER inferred
- Per-head blacklist is enforced
- Supports --dry run
"""

from pathlib import Path
import argparse
from typing import Dict

import fasttext
from tqdm import tqdm

from lib import eebo_db
from lib import eebo_config as config
from lib.eebo_logging import logger


# ---------- helpers ----------

def load_fasttext_model(model_path: Path):
    logger.info(f"Loading global fastText model from {model_path}")
    return fasttext.load_model(str(model_path))


def is_reasonable_orthographic_variant(candidate: str, canonical: str) -> bool:
    """
    Conservative orthographic filter.
    Allows OCR / early modern variation, blocks semantic drift.
    """

    if candidate == canonical:
        return False

    # length sanity
    if abs(len(candidate) - len(canonical)) > 3:
        return False

    # stem overlap heuristic
    if canonical[:4] not in candidate and candidate[:4] not in canonical:
        return False

    # avoid negation prefixes collapsing concepts
    for bad_prefix in ("un", "in", "dis", "non"):
        if candidate.startswith(bad_prefix) and not canonical.startswith(bad_prefix):
            return False

    return True


def insert_spelling_map(conn, variant: str, canonical: str, dry: bool):
    if dry:
        logger.debug(f"[DRY] spelling_map: {variant} -> {canonical}")
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO spelling_map (variant, canonical, concept_type)
            VALUES (%s, %s, 'orthographic')
            ON CONFLICT (variant)
            DO UPDATE SET canonical = EXCLUDED.canonical;
            """,
            (variant, canonical),
        )


def update_tokens_canonical(conn, variant: str, canonical: str, dry: bool):
    if dry:
        logger.debug(f"[DRY] tokens.canonical: {variant} -> {canonical}")
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE tokens
            SET canonical = %s
            WHERE token = %s;
            """,
            (canonical, variant),
        )


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Build spelling_map using fixed canonical heads"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(config.FASTTEXT_GLOBAL_MODEL_PATH),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.TOP_K,
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Dry run: no database writes",
    )

    args = parser.parse_args()

    if args.dry:
        logger.info("DRY RUN enabled")

    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    model = load_fasttext_model(model_path)

    keyword_map: Dict[str, set[str]] = config.KEYWORDS_TO_NORMALISE
    if not keyword_map:
        logger.error("KEYWORDS_TO_NORMALISE is empty")
        return

    report = []

    with eebo_db.get_connection(application_name="spelling_map_builder") as conn:

        for canonical, blacklist in tqdm(
            keyword_map.items(),
            desc="Canonical heads processed",
        ):
            try:
                neighbors = model.get_nearest_neighbors(canonical, args.top_k)
            except KeyError:
                logger.warning(f"Canonical '{canonical}' not in fastText vocab")
                continue

            for score, candidate in neighbors:
                candidate = candidate.lower()

                if candidate in blacklist:
                    continue

                if not is_reasonable_orthographic_variant(candidate, canonical):
                    continue

                insert_spelling_map(conn, candidate, canonical, args.dry)
                update_tokens_canonical(conn, candidate, canonical, args.dry)

                report.append((canonical, candidate, score))

        if not args.dry:
            conn.commit()
            logger.info("Database updated")

    # reporting
    logger.info("Canonicalisation report:")
    for canonical, variant, score in report:
        logger.info(f"{variant} -> {canonical} ({score:.3f})")

    logger.info("Done.")


if __name__ == "__main__":
    main()
