from pathlib import Path
from lib.eebo_config import CORPUS_TIER2_DB_PATH, SLICES
from lib.eebo_logging import logger

CORPUS_START_YEAR = SLICES[0]
CORPUS_END_YEAR = SLICES[-1]

output_path = Path(GUI_PUBLIC_ROOT) / "src" / "corpus_config.ts"

content = f"""
export const CORPUS_TIER2_DB_PATH = "{CORPUS_TIER2_DB_PATH}";
export const CORPUS_START_YEAR = {CORPUS_START_YEAR};
export const CORPUS_END_YEAR = {CORPUS_END_YEAR};
""".lstrip()

output_path.write_text(content, encoding="utf-8")

logger.info(f"Wrote {output_path}")
