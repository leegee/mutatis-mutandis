# lib/eebo_config.py

from pathlib import Path
from typing import TypedDict, Set, Dict


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EEBO_SRC_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "eebo_all" / "eebo_phase1" / "P4_XML_TCP"

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

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_DIR = OUT_DIR / "faiss"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_CANONICAL_INDEX_PATH = INDEXES_DIR / "canonical.faiss.bin"
FAISS_CANONICAL_INDEX_PATH = FAISS_INDEX_DIR / "canonical.faiss.bin"

MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")
FASTTEXT_GLOBAL_MODEL_PATH = MODELS_DIR / "global_eebo.bin"
FASTTEXT_SLICE_MODEL_DIR = MODELS_DIR / "fastTextCanonSlice"
BATCH_DOCS = 100
BATCH_TOKENS = 10000
FASTTEXT_BATCH_SIZE = 50_000
EMBED_BATCH_SIZE = 256
INGEST_TOKEN_WINDOW_FALLBACK = 5  # around 5 tokens if sentence unavailable
NUM_WORKERS = 4
CANONICALISATION_BATCH_SIZE = 50_000

STOPWORD_FILE = EEBO_SRC_DIR / "stopwords" / "english_basic.txt"
TOP_K = 30

FASTTEXT_PARAMS = {
    "model": "skipgram",      # Skip-gram model
    "dim": 100,               # Word vector dimensionality - 200
    "epoch": 5,               # Number of epochs           - 10
    "ws": 5,                  # Context window size
    "minCount": 1,            # Keep all words
    "thread": 6,              # Adjust to for this CPU
    "minn": 2,                # Subword ngram min length
    "maxn": 5,                # Subword ngram max length
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

# Absorption class thresholds
STRONG_MEAN = 0.80
MODERATE_MEAN = 0.65
MIN_SLICES_STRONG = 3

"""
Canonical normalisation configuration.

KEYWORDS_TO_NORMALISE is now the SINGLE source of truth.

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
    allowed_variants: Set[str]
    false_positives: Set[str]

CanonicalRules = Dict[str, CanonicalRule]

# Canonical heads with per-head exclusion lists
KEYWORDS_TO_NORMALISE: CanonicalRules = {
    "justice": {
        "allowed_variants": {
            "unjustice",
            "vnjustice",
            "dinjustice",
            "iujustice",
            "chiefjustice",
            "executejustice",
            "satisfiedjustice",
        },
        "false_positives": {
            "injury",
            "injustice",
        },
    },
    "injustice": {
        "allowed_variants": {
            "dojustice",
            "ofjustice",
        },
        "false_positives": {
            "injury",
            "justice",
        },
    },
    "liberty": {
        "allowed_variants": {
            "afreedom",
            "freeliberty",
            "bufreedom",
            "freedomship",
        },
        "false_positives": set(),
    },
    "freedom": {
        "allowed_variants": {
            "liberty",
            "afreedom",
            "bufreedom",
        },
        "false_positives": set(),
    },
    "reasonable": {
        "allowed_variants": {
            "ureasonable",
            "unreasonable",
        },
        "false_positives": set(),
    },
    "state": {
        "allowed_variants": {
            "estate",
        },
        "false_positives": set(),
    },
}

