# lib/corpus_config.py

import os
import sys
from pathlib import Path

# from lib.corpus_pamphlets import CORPUS_MIN_YEAR, CORPUS_MAX_YEAR, CONCEPT_SETS, PHRASE_SETS, FILTER_DOCUMENT_SIZE, MIN_TOKENS_IN_DOC, MAX_TOKENS_IN_DOC
from lib.corpus_albinism import get_lance_indexes_dir, CORPUS_MIN_YEAR, CORPUS_MAX_YEAR, CONCEPT_SETS, PHRASE_SETS, FILTER_DOCUMENT_SIZE, MIN_TOKENS_IN_DOC, MAX_TOKENS_IN_DOC

# CORPUS_MIN_YEAR = 1600
# CORPUS_MAX_YEAR = 1650
# FILTER_DOCUMENT_SIZE = False
# MIN_TOKENS_IN_DOC = 200
# MAX_TOKENS_IN_DOC = 400_000


EMBED_BATCH_SIZE = 64 # Tests showed is faster by ~30% than 256
TOP_K = 30

SCALES = ("local", "medium", "broad")
ACTIVE_SCALES = ("local")

WINDOW_CONFIGS = (
    {"name": "local", "size": 30, "stride": 20},
    {"name": "medium", "size": 256, "stride": 128},
    {"name": "broad", "size": 512, "stride": 384},
)
SCALE_NAMES = SCALES # tuple(config["name"] for config in WINDOW_CONFIGS)

LANCE_MODEL_NAME = "macberth"
LANCE_BUCKET_SIZE = 50


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ECCO_HEADER_DIR = Path( PROJECT_ROOT / "corpus/ecco_all/ecco/headers" )

# CORPUS_ROOT_DIR = PROJECT_ROOT / "corpus"
CORPUS_ROOT_DIR = Path("G:/") / "corpus"

CORPUS_INPUT_DIRS = {
    "misc": CORPUS_ROOT_DIR / "misc",
    "eebo": CORPUS_ROOT_DIR / "eebo_all",
    "ecco": CORPUS_ROOT_DIR / "ecco_all",
}

CLMET_CORPUS_INPUT_DIR = CORPUS_ROOT_DIR / "clmet" / "corpus" / "txt" / "plain"

COLAB_MODE = (
    os.environ.get("COLAB_MODE", "").lower() in ("1", "true", "yes")
    or "google.colab" in sys.modules
)

OUT_DIR = Path("/content/drive/MyDrive/macberth_output") if COLAB_MODE else PROJECT_ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAPPED_GOOGLE_DRIVE = Path("E:") / "My Drive" / "passage-pipeline"
MAPPED_GOOGLE_DRIVE.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

GUI_PUBLIC_ROOT      = Path(PROJECT_ROOT / 'gui' / 'eebo-frontend' )
GUI_PUBLIC_DIR       = Path(GUI_PUBLIC_ROOT / 'public')
GUI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_TIER2_DB_URL        = str(Path('data') / 'tier2_concept_neighbours.db')
CORPUS_TIER2_MASKED_DB_URL = str(Path('data') / 'tier2_concept_neighbours_MASKED.db')

CORPUS_TIER2_DB_PATH        = Path(GUI_PUBLIC_DIR / CORPUS_TIER2_DB_URL)
CORPUS_TIER2_MASKED_DB_PATH = Path(GUI_PUBLIC_DIR / CORPUS_TIER2_MASKED_DB_URL)

CORPUS_TIER3_DB_URL        = str(Path('data') / 'tier3_concept_neighbours.db')
CORPUS_TIER3_MASKED_DB_URL = str(Path('data') / 'tier3_concept_neighbours_MASKED2.db')

CORPUS_TIER3_DB_PATH        = Path(GUI_PUBLIC_DIR / CORPUS_TIER3_DB_URL)
CORPUS_TIER3_MASKED_DB_PATH = Path(GUI_PUBLIC_DIR / CORPUS_TIER3_MASKED_DB_URL)

JOBS_DB_PATH = OUT_DIR / "fastapi_jobs.sqlite3"

INDEXES_DIR = OUT_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

# LANCE_INDEXES_DIR = INDEXES_DIR / "lance_albinism"
# LANCE_INDEXES_DIR = INDEXES_DIR / "lance_pamphlets"
LANCE_INDEXES_DIR = get_lance_indexes_dir(INDEXES_DIR)
LANCE_INDEXES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = GUI_PUBLIC_DIR / "data" / "scatter"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

