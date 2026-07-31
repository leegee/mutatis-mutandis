from pathlib import Path

from lib.eebo_config import GUI_PUBLIC_ROOT, CORPUS_TIER2_DB_URL, CORPUS_TIER2_MASKED_DB_URL
from lib.corpus_logging import logger
from lib.get_corpus_year_range import get_corpus_year_range


CORPUS_START_YEAR, CORPUS_END_YEAR = get_corpus_year_range()

if CORPUS_START_YEAR is None or CORPUS_END_YEAR is None:
    raise RuntimeError("Could not determine corpus year range from database")


output_path = Path(GUI_PUBLIC_ROOT) / "src" / "corpus_config.ts"

content = f"""
export const CORPUS_START_YEAR = {CORPUS_START_YEAR};
export const CORPUS_END_YEAR = {CORPUS_END_YEAR};
export const CORPUS_TIER2_DB_URL = "/{Path(CORPUS_TIER2_DB_URL).as_posix()}";
""".lstrip()

output_path.write_text(content, encoding="utf-8")

logger.info(
    f"Wrote {output_path} with corpus range {CORPUS_START_YEAR}-{CORPUS_END_YEAR}"
)
