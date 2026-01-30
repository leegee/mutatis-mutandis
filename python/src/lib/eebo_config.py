# lib/eebo_config.py

from pathlib import Path
from typing import TypedDict, Set, Dict

class FastTextParams(TypedDict):
    model: str
    dim: int
    epoch: int
    ws: int
    minCount: int
    thread: int
    minn: int
    maxn: int

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EEBO_SRC_DIR = Path(__file__).resolve().parent
XML_ROOT_DIR = BASE_DIR / "eebo_all" / "eebo_phase1" / "P4_XML_TCP"

# Should use a dict for this:
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

PLAIN_DIR = OUT_DIR / "plain"
PLAIN_DIR.mkdir(parents=True, exist_ok=True)

SLICES_DIR = OUT_DIR / "slices"
SLICES_DIR.mkdir(parents=True, exist_ok=True)

INDEXES_DIR = OUT_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FASTTEXT_SLICE_MODEL_DIR = MODELS_DIR / "fastTextSlices"

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_DIR = OUT_DIR / "faiss"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")

BATCH_DOCS = 100
BATCH_TOKENS = 10000
FASTTEXT_BATCH_SIZE = 50_000
EMBED_BATCH_SIZE = 256
INGEST_TOKEN_WINDOW_FALLBACK = 5  # around 5 tokens if sentence unavailable
NUM_WORKERS = 4

STOPWORD_FILE = EEBO_SRC_DIR / "stopwords" / "english_basic.txt"
TOP_K = 30

QUICKIE_FASTTEXT_PARAMS: FastTextParams = {
    "model": "skipgram",      # Skip-gram model
    "dim": 100,               # Word vector dimensionality - 200
    "epoch": 5,               # Number of epochs           - 10
    "ws": 5,                  # Context window size        - 10
    "minCount": 1,            # Keep all words
    "thread": 6,              # Adjust to for this CPU
    "minn": 2,                # Subword ngram min length   - 3
    "maxn": 5,                # Subword ngram max length   - 6
}

FASTTEXT_PARAMS: FastTextParams = {
    "model": "skipgram",
    "dim": 200,
    "epoch": 10,
    "ws": 7,
    "minCount": 1,
    "thread": 6,
    "minn": 3,
    "maxn": 6,
}

SLICES = [
    (1625, 1629),
    (1630, 1634),
    (1635, 1639),
    (1640, 1640),
    (1641, 1641),
    (1642, 1642),
    (1643, 1643),
    (1644, 1644),
    (1645, 1645),
    (1646, 1646),
    (1647, 1647),
    (1648, 1648),
    (1649, 1649),
    (1650, 1650),
    (1651, 1651),
    (1652, 1654),
    (1655, 1657),
    (1658, 1660),
    (1661, 1665),
]

"""
Canonical normalisation configuration.

CONCEPT_SETS is now the SINGLE source of truth.

- dict keys: canonical heads (theory-driven)
- dict values:
    - allowed_variants: forms that may be normalised *to* this head
    - false_positives: forms that must never be normalised to this head, even if
      they are close in spelling or embedding space.

Normalisation is restricted to orthographic- and boundary-level variation characteristic
of early modern print and OCR, including the collapse of whitespace between function words
and lexical heads (eg `ofjustice`). These forms are treated as recoverable tokenisation
artefacts rather than distinct lexical items. Semantic distinctions between canonical
concepts are preserved through explicit constraints, positive and negative, on allowable mappings.

"""
class CanonicalRule(TypedDict):
    forms: Set[str]
    false_positives: Set[str]

CanonicalRules = Dict[str, CanonicalRule]

# Canonical heads with per-head exclusion lists
# liberty
# authority
# sovereignty
# obedience
# law
# parliament
# king
# people
# commonwealth
# tyranny
# conscience
# religion
# church
# state
# power
# right
# property
CONCEPT_SETS: CanonicalRules = {
    "LIBERTY": {
        "forms": {
            "liberty", "libertie", "libertye", "liberte",
            "libertyes", "libert", "liberties",
            "liliberty", "libertv", "libertty", "lyliberty",
            "libery", "libertly", "fulliberty", "lilibertyis",
            "thliberties", "liberry", "libertyby",
            "iberty", "libertle", "libertles", "libertys",
            "iiberty", "iberties", "libety", "liberts",
            "libertyliberty", "libertynow", "libertees",
            "libertee", "libertes"
        },
        "false_positives": {
            "libertine", "libertin", "libertins", "libertinage",
            "libertind", "libertyin", "liberality"
        }
    },

    "PROPERTY": {
        "forms": {
            "property", "propertie", "propriety"
        },
        "false_positives": set()
    }
}

