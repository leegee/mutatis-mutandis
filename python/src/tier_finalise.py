from lib.eebo_config import CORPUS_TIER2_DB_PATH, SLICES

CORPUS_START_YEAR = SLICES[0]
CORPUS_END_YEAR   = SLICES[length(SLICES) - 1] # TODO Pythonise

# Open and write to GUI_PUBLIC_ROOT / "src" / "corpus_config.ts"

print(f"""

export const CORPUS_TIER2_DB_URL = "{CORPUS_TIER2_DB_URL}";
export const CORPUS_START_YEAR = {CORPUS_START_YEAR};
export const CORPUS_END_YEAR = {CORPUS_END_YEAR};

""")


