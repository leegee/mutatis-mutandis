# lib/eebo_config.py

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EEBO_SRC_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "eebo_all" / "eebo_phase1" / "P4_XML_TCP"
OUT_DIR = BASE_DIR / "out"
TMP_DIR = BASE_DIR / "tmp"
PLAIN_DIR = OUT_DIR / "plain"
SLICES_DIR = OUT_DIR / "slices"
SLICES_DIR = BASE_DIR / "out" / "slices"
MODELS_DIR = BASE_DIR / "out" / "models"

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

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

QUERY_FILE = EEBO_SRC_DIR / "queries" / "canonical.txt"
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

- dict keys   → canonical heads (theory-driven)
- dict values → explicit blacklist of tokens that must NEVER map to that head

Normalisation is restricted to orthographic and boundary-level variation characteristic
of early modern print and OCR, including the elision of whitespace between function words
and lexical heads (e.g. ofjustice). These forms are treated as recoverable tokenisation
artefacts rather than distinct lexical items. Semantic distinctions between canonical
concepts are preserved through explicit constraints on allowable mappings.

"""

# Canonical heads with per-head exclusion lists
KEYWORDS_TO_NORMALISE: dict[str, set[str]] = {
    "justice": {
        "injustice",
        "injury",
        "unjustice",
        "vnjustice",
        "dinjustice",
        "iujustice",
        "chiefjustice",
        "executejustice",
        "satisfiedjustice",
    },
    "injustice": {
        "justice",
        "dojustice",
        "ofjustice",
    },
    "liberty": {
        "afreedom",
        "freeliberty",
        "bufreedom",
        "freedomship",
    },
    "freedom": {
        "liberty",
        "afreedom",
        "bufreedom",
    },
    "reasonable": {
        "ureasonable",
        "unreasonable",
    },
    "state": {
        "estate",
    },
}

