from pathlib import Path
from lib.eebo_config import GUI_PUBLIC_ROOT, CORPUS_TIER2_DB_URL, CORPUS_TIER2_MASKED_DB_URL, SLICES
from lib.eebo_logging import logger

CORPUS_START_YEAR = SLICES[0][0]
CORPUS_END_YEAR = SLICES[-1][-1]

output_path = Path(GUI_PUBLIC_ROOT) / "src" / "corpus_config.ts"

content = f"""
export const CORPUS_START_YEAR = {CORPUS_START_YEAR};
export const CORPUS_END_YEAR = {CORPUS_END_YEAR};
// export const CORPUS_TIER2_DB_URL = /"{Path(CORPUS_TIER2_MASKED_DB_URL).as_posix()}";
export const CORPUS_TIER2_DB_URL = /"{Path(CORPUS_TIER2_DB_URL).as_posix()}";
""".lstrip()

output_path.write_text(content, encoding="utf-8")

logger.info(f"Wrote {output_path}")
