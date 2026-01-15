from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EEBO_SRC_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "eebo_all" / "eebo_phase1" / "P4_XML_TCP"
OUT_DIR = BASE_DIR / "out"
PLAIN_DIR = OUT_DIR / "plain"
SLICES_DIR = OUT_DIR / "slices"

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = Path("./lib/macberth-huggingface")
INGEST_BATCH_SIZE = 20_000
EMBED_BATCH_SIZE = 256
INGEST_TOKEN_WINDOW_FALLBACK = 5  # around 5 tokens if sentence unavailable
EMBED_MAX_WORKERS = 4

SLICES_DIR = BASE_DIR / "out" / "slices"
MODELS_DIR = BASE_DIR / "out" / "models"

QUERY_FILE = EEBO_SRC_DIR / "queries" / "canonical.txt"
STOPWORD_FILE = EEBO_SRC_DIR / "stopwords" / "english_basic.txt"
TOP_K = 30

FASTTEXT_PARAMS = {
    "model": "skipgram",      # Skip-gram model
    "dim": 200,               # Word vector dimensionality
    "ws": 5,                  # Context window size
    "epoch": 10,              # Number of epochs
    "minCount": 1,            # Keep all words
    "thread": 4,              # Adjust to for this CPU
    "minn": 3,                # Subword ngram min length
    "maxn": 6,                # Subword ngram max length
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
