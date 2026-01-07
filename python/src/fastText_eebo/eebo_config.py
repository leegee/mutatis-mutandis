from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EEBO_SRC_DIR = Path(__file__).resolve().parent

print(f"[INFO] BASE_DIR: {BASE_DIR}")
print(f"[INFO] EEBO_SRC_DIR: {EEBO_SRC_DIR}")

INPUT_DIR = BASE_DIR / "eebo_all" / "eebo_phase1" / "P4_XML_TCP"
OUT_DIR = BASE_DIR / "out"
PLAIN_DIR = OUT_DIR / "plain"
SLICES_DIR = OUT_DIR / "slices"
DB_PATH = OUT_DIR / "metadata.sqlite"

SLICES_DIR = BASE_DIR / "out" / "slices"
MODELS_DIR = BASE_DIR / "out" / "models"

QUERY_FILE = EEBO_SRC_DIR / "queries" / "canonical.txt"
STOPWORD_FILE = EEBO_SRC_DIR / "stopwords" / "english_basic.txt"
TOP_K = 30
