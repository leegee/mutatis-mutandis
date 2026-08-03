from pathlib import Path

from lib.corpus_db import get_connection
from lib.corpus_config import GUI_PUBLIC_ROOT, CORPUS_TIER2_DB_URL, CORPUS_TIER2_MASKED_DB_URL
from lib.corpus_logging import logger
from lib.get_corpus_year_range import get_corpus_year_range

output_path = Path(GUI_PUBLIC_ROOT) / "src" / "corpus_config.ts"

CORPUS_START_YEAR, CORPUS_END_YEAR = get_corpus_year_range()
if CORPUS_START_YEAR is None or CORPUS_END_YEAR is None:
    raise RuntimeError("Could not determine corpus year range from database")

CORPUS_START_YEAR = 1620

pg_conn = get_connection()

try:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        total_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM tokens")
        total_tokens = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pamphlet_corpus")
        total_corpus_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM pamphlet_tokens")
        total_corpus_tokens = cur.fetchone()[0]
finally:
    pg_conn.close()


content = f"""
export const CORPUS_START_YEAR = {CORPUS_START_YEAR};
export const CORPUS_END_YEAR = {CORPUS_END_YEAR};
export const CORPUS_TIER2_DB_URL = "/{Path(CORPUS_TIER2_DB_URL).as_posix()}";

export const CORPUS_COUNTS = {{
    total_docs: {total_docs},
    total_tokens: {total_tokens},
    total_corpus_docs: {total_corpus_docs},
    total_corpus_tokens: {total_corpus_tokens}
}};
""".lstrip()

output_path.write_text(content, encoding="utf-8")

logger.info(
    f"Wrote {output_path} with corpus range {CORPUS_START_YEAR}-{CORPUS_END_YEAR}"
)
