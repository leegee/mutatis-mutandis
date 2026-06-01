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
XML_ROOT_DIR = BASE_DIR / "eebo_all"

try:
    import google.colab  # noqa: F401
    COLAB_MODE = True
except ModuleNotFoundError:
    COLAB_MODE = False

# Could use env var
OUT_DIR = Path("/content/drive/MyDrive/macberth_output") if COLAB_MODE else BASE_DIR / "out"
print(f"OUT_DIR = {OUT_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

SQLITE_DB_PATH     = Path(OUT_DIR / '..' / 'gui' / 'eebo-frontend' / 'public' / 'data' / 'tier2_concept_neighbours.db')

INDEXES_DIR = OUT_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

ZARR_ROOT = OUT_DIR / "zarr"

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MACBERTH_MODEL_PATH = Path("./lib/macberth-huggingface")
MACBERTH_MODEL_NAME = "emanjavacas/MacBERTh"

FAISS_INDEX_DIR = INDEXES_DIR / "faiss"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_TIER1_INDEX = FAISS_INDEX_DIR / "tier1.index"
FAISS_SLICE_DIR = FAISS_INDEX_DIR / "slices"

BATCH_DOCS = 100
BATCH_TOKENS = 10000
FASTTEXT_BATCH_SIZE = 50_000
EMBED_BATCH_SIZE = 256
INGEST_TOKEN_WINDOW_FALLBACK = 5  # around 5 tokens if sentence unavailable
NUM_WORKERS = 4

STOPWORD_FILE = EEBO_SRC_DIR / "stopwords" / "english_basic.txt"
TOP_K = 30

# For now, mirror in JSON file gui/eebo-frontend/corpus_config.ts
# These are no longer used.
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

For FT, normalisation is restricted to orthographic- and boundary-level variation characteristic
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
    "PREROGATIVE": {
        "forms": {
            "prerogative", "prerogatiue",
            "prerogatives", "prerogatiues",
        },
        "false_positives": set(),
    },
    "LAW": {
        "forms": {
            "law", "laws", "lawe", "lawes",
        },
        "false_positives": {
            "clawes", "claw", "flaw", "lawne",
            "thlaw", # welsh
            "laz", "layt", "layen",
            },
    },
    "LIBERTY": {
        "forms": {
            "liberty",
            "libe_ty", "liberry",
            "libertie", "libertye", "liberte",
            "liliberty", "libertv", "libertty", "lyliberty",
            "libertyes", "libert", "liberties",
            "libery", "libertly", "fulliberty", "lilibertyis",
            "thliberties", "libertyby",
            "iberty", "libertle", "libertles", "libertys",
            "iiberty", "iberties", "libety", "liberts",
            "libertynow", "libertees", "libetty",
            "libertee", "libertes", "lyberty",
            "lberty", "libertis", "leberty",
            "liberrie", "lliberty"

            # Keep for now then remove after re-ingestion which will normalise:
            "aliberty", "liberti", "berty",
            # Need to pre-ingest fix "lib erty" etc
            "generalliberty",
            "understandingliberty",
            "libe",
            # Diagnostic only, remove later:
            # "liber",
        },
        "false_positives": {
            "libertine", "libertin", "libertins", "libertinage", "libertinism",
            "libertind", "libertyin", "liberality",  "libertinisme", "liberallity",
            "libertism", "libertines", "liberta",
            # "liberal", "liberall",
            "libels",
            "libertate", "libertates", "liberabit", "libero","deliberates", "deliberated",
            "liberto","liberabo","liberall","liberalytie","liberaui","liberally","liberates","liberalitie",
            "libera", "deliberate", "liberando", "libya", "liberi", "liberior"

        },
    },


    "DIVINE": {
        "forms": {
            "god", "divine", "heaven", "heavens", "eternal", "grace", "providence",
            "sacred", "holy", "lord", "almighty", "creator", "eternity"
        },
        "false_positives": {
            "godly", "good", "goods", "gold", "glad"
        }
    },
    "TEMPORAL": {
        "forms": {
            "state", "civil", "political", "temporal", "commonwealth",
            "kingdom", "government", "authority", "prince", "realm"
        },
        "false_positives": {
            "statue", "station", "temple"
        }
    },


    # Rough political/legal
    "KING": {
        "forms": {"king", "kings", "kinges", "monarch", "sovereign"},
        "false_positives": {"kin", "kine", "sink", "sing"}
    },
    "PARLIAMENT": {
        "forms": {"parliament", "parliment", "parliaments"},
        "false_positives": {"parliamentary", "parlour"}
    },
    "OBEDIENCE": {
        "forms": {"obedience", "obedient", "obedienc", "obey"},
        "false_positives": {"obscene", "obeyed", "obed"}
    },
    "PEOPLE": {
        "forms": {"people", "peoples", "peple", "populace", "subjects"},
        "false_positives": {"peep", "peeps", "pepla"}
    },
    "COMMONWEALTH": {
        "forms": {"commonwealth", "common-wealth", "common weal"},
        "false_positives": {"common", "wealth"}
    },

    # Rough theology
    "CHURCH": {
        "forms": {"church", "churches", "clergy", "ecclesia", "congregation"},
        "false_positives": {"churchyard", "churchman"}
    },
    "RELIGION": {
        "forms": {"religion", "religions", "faith", "doctrine", "creed"},
        "false_positives": {"religious", "religionist"}
    },

    # Neutral baselines
    "MAN": {
        "forms": {"man"},
        "false_positives": {"woman"},
    },

    "HOUSE": {
        "forms": {"house"},
        "false_positives": {},
    },

    "PROPERTY": {
        "forms": {
            "property", "propertie", "propriety"
        },
        "false_positives": {
            "properly"
        }
    },

    # May 2026
    "REVOLUTION": {
        "forms": {
            "revolution", "revolucion", "revolutio", "revolutions", "revolutión",
            "revolucon", "revolucionary", "revolucioners", "revolutioners",
            "rebellion", "insurrection"  # Often semantically overlapping in period usage
        },
        "false_positives": {
            "revolution" : ["astronomical", "planetary", "celestial", "orb", "circle"]  # Pre-1688, often means literal 'turning/rotation'
        }
    },
    "INTEREST": {
        "forms": {
            "interest", "interesse", "intrest", "intrests", "interests",
            "interestes", "interessed", "interessing", "publique interest",
            "common interest", "particular interest"
        },
        "false_positives": set(),
        # { "usury", "usance", "money", "profit", "compound" }
    },
    "FANATIC": {
        "forms": {
            "fanatic", "fanatick", "fanatique", "fanaticks", "fanatiques",
            "fanaticism", "fanaticisme", "phanatic", "phanatique"
        },
        "false_positives": set(),
    },
    "ENTHUSIASM": {
        "forms": {
            "enthusiasm", "enthusiasme", "enthousiasm", "enthusiast", "enthusiasts",
            "enthusiastick", "enthusiastical", "enthusiasms", "enthusiastical"
        },
        "false_positives": set(),
    }

}

