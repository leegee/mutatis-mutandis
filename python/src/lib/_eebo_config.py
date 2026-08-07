# lib/corpus_config.py

from pathlib import Path
from typing import TypedDict, Set, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
XML_ROOT_DIR = PROJECT_ROOT / "corpus"

CORPUS_INPUT_DIRS = {
    "eebo": XML_ROOT_DIR / "eebo_all",
    "ecco": XML_ROOT_DIR / "ecco_all",
}

ECCO_HEADER_DIR = Path( "s:/src/pamphlets/corpus/ecco_all/ecco/headers" )

try:
    import google.colab  # noqa: F401
    COLAB_MODE = True
except ModuleNotFoundError:
    COLAB_MODE = False

# It would be good to have some large docs (A91273)
# but complete Bibles maybe oveor-balance the index?
MIN_TOKENS_IN_DOC = 200
# MAX_TOKENS_IN_DOC = 20000 # safe
MAX_TOKENS_IN_DOC = 350000 # takes > 24 hours on my PC

# Could use env var
OUT_DIR = Path("/content/drive/MyDrive/macberth_output") if COLAB_MODE else PROJECT_ROOT / "out"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TMP_DIR = OUT_DIR / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

GUI_PUBLIC_ROOT      = Path(PROJECT_ROOT / 'gui' / 'eebo-frontend' )
GUI_PUBLIC_DIR       = Path(GUI_PUBLIC_ROOT / 'public')
GUI_PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_TIER2_DB_URL        = str(Path('data') / 'tier2_concept_neighbours.db')
CORPUS_TIER2_MASKED_DB_URL = str(Path('data') / 'tier2_concept_neighbours_MASKED.db')

CORPUS_TIER2_DB_PATH        = Path(GUI_PUBLIC_DIR / CORPUS_TIER2_DB_URL)
CORPUS_TIER2_MASKED_DB_PATH = Path(GUI_PUBLIC_DIR / CORPUS_TIER2_MASKED_DB_URL)

JOBS_DB_PATH = OUT_DIR / "fastapi_jobs.sqlite3"

INDEXES_DIR = OUT_DIR / "indexes"
INDEXES_DIR.mkdir(parents=True, exist_ok=True)

ZARR_ROOT = OUT_DIR / "zarr"
ZARR_ROOT.mkdir(parents=True, exist_ok=True)

ZARR_PATH = ZARR_ROOT / "tier1"
ZARR_PATH.mkdir(parents=True, exist_ok=True)

MASKED_ZARR_PATH = ZARR_ROOT / "tier1-masked"
MASKED_ZARR_PATH.mkdir(parents=True, exist_ok=True)

MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = OUT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_DIR = INDEXES_DIR / "faiss"
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
FAISS_TIER1_INDEX        = FAISS_INDEX_DIR / "tier1.index"
FAISS_TIER1_INDEX_MASKED = FAISS_INDEX_DIR / "tier1-masked.index"

FAISS_SCALES = ("local", "medium", "broad")

# TODO Move
def faiss_index_paths(masked: bool, year: int | None = None) -> dict[str, Path]:
    """
    e.g. faiss_index_paths(masked=True)         -> tier1_local_masked.faiss
         faiss_index_paths(masked=True, year=1625) -> tier1_local_1625_masked.faiss

    Year is inserted before the masked suffix so that all files for a given
    scale still sort/group together regardless of masked status.
    """
    year_part = f"_{year}" if year is not None else ""
    masked_part = "_masked" if masked else ""
    return {
        scale: FAISS_INDEX_DIR / f"tier1_{scale}{year_part}{masked_part}.faiss"
        for scale in FAISS_SCALES
    }

# TODO Move
def discover_index_years(masked: bool) -> list[int]:
    """
    Find which years already have on-disk FAISS indices, by globbing for
    files matching the tier1_medium_<year>[_masked].faiss naming pattern.
    """
    import re
    medium_base = faiss_index_paths(masked)["medium"]
    masked_part = "_masked" if masked else ""
    pattern = f"tier1_medium_*{masked_part}.faiss"
    year_re = re.compile(rf"^tier1_medium_(\d+){masked_part}\.faiss$")

    years = []
    for p in medium_base.parent.glob(pattern):
        m = year_re.match(p.name)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


PLOT_DIR = GUI_PUBLIC_DIR / "data" / "scatter"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_BATCH_SIZE = 256

TOP_K = 30

CORPUS_MIN_YEAR = 1625 # 1600
CORPUS_MAX_YEAR = 1689 # 1800

# These are no longer used.
OLD_SLICES = [
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
    (1661, 1664),
    (1665, 1669),
    (1670, 1674),
    (1675, 1679),
    (1680, 1684),
    (1685, 1689),
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
TEST_CONCEPT_SETS: CanonicalRules = {
    "PREROGATIVE": {
        "forms": {
            "prerogative",
        },
        "false_positives": set(),
    },
}

CONCEPT_SETS: CanonicalRules = {
    "PREROGATIVE": {
        "forms": {
            "prerogative", # "prerogatiue", "prerogatives", "prerogatiues",
        },
        "false_positives": set(),
    },
    "LAW": {
        "forms": {
            "law", # "laws", "lawe", "lawes",
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
            # "libe_ty", "liberry", "libertie", "libertye", "liberte", "liliberty", "libertv", "libertty", "lyliberty", "libertyes", "libert", "liberties", "libery", "libertly", "fulliberty", "lilibertyis", "thliberties", "libertyby", "iberty", "libertle", "libertles", "libertys", "iiberty", "iberties", "libety", "liberts", "libertynow", "libertees", "libetty", "libertee", "libertes", "lyberty", "lberty", "libertis", "leberty", "liberrie", "lliberty"

            # Keep for now then remove after re-ingestion which will normalise:
            # "aliberty", "liberti", "berty",
            # Need to pre-ingest fix "lib erty" etc
            # "generalliberty",
            # "understandingliberty",
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
            "divine",
            # "god",  "heaven", "heavens", "eternal", "grace", "providence", "sacred", "holy", "lord", "almighty", "creator", "eternity"
        },
        "false_positives": {
             "good", "goods", "gold", "glad" # "godly",
        }
    },
    "TEMPORAL": {
        "forms": {
            "temporal", # "state", "civil", "political",  "commonwealth", "kingdom", "government", "authority", "prince", "realm"
        },
        "false_positives": set(), # { "statue", "station", "temple" }
    },


    # Rough political/legal
    "KING": {
        "forms": {"king", }, # "kings", "kinges",  # "monarch", "sovereign"
        "false_positives": {"kin", "kine", "sink", "sing"}
    },
    "PARLIAMENT": {
        "forms": {"parliament", }, # "parliment", "parliaments"
        "false_positives": { "parlour"} # "parliamentary",
    },
    "OBEDIENCE": {
        "forms": {"obedience"}, # , "obedient", "obedienc", "obey"
        "false_positives": {"obscene", "obeyed", "obed"}
    },
    "PEOPLE": {
        "forms": {"people"}, #  "peoples", "peple",   "populace", "subjects"
        "false_positives": {"peep", "peeps", "pepla"}
    },
    "COMMONWEALTH": {
        "forms": {"commonwealth", "common-wealth", "common weal"},
        "false_positives": {"common", "wealth"}
    },

    # Rough theology
    "CHURCH": {
        "forms": {"church"}, # , "churches", "clergy", "ecclesia", "congregation"
        "false_positives": set(), # {"churchyard", "churchman"}
    },
    "RELIGION": {
        "forms": {"religion"}, # , "religions", "faith", "doctrine", "creed"
        "false_positives": set(),  # {"religious", "religionist"}
    },

    # Neutral baselines
    "MAN": {
        "forms": {"man"},
        "false_positives": set()  # {"woman"},
    },

    "HOUSE": {
        "forms": {"house"},
        "false_positives": {},
    },

    "PROPERTY": {
        "forms": {
            "property", #  "propertie", "propriety"
        },
        "false_positives": set() # { "properly" }
    },

    # May 2026
    "REVOLUTION": {
        "forms": {
            "revolution" # , "revolucion", "revolutio", "revolutions", "revolutión", "revolucon", "revolucionary", "revolucioners", "revolutioners",
            # "rebellion", "insurrection"
        },
        "false_positives": set(), # { "astronomical", "planetary", "celestial", "orb", "circle" }
    },
    "INTEREST": {
        "forms": {
            "interest", # "interesse", "intrest", "intrests", "interests", "interestes", "interessed",
        },
        "false_positives": set(),
        # { "usury", "usance", "money", "profit", "compound" }
    },
    "FANATIC": {
        "forms": {
            "fanatic", #  "fanatick", "fanatique", # "fanaticism", "fanaticisme", "phanatic", "phanatique"
        },
        "false_positives": set(),
    },
    "ANABAPTIST": {
        "forms": {"anabaptist"},
        "false_positives": set(),
    },
    "ENTHUSIASM": {
        "forms": {
            "enthusiasm",
            # "enthusiasme", "enthousiasm", "enthusiast", "enthusiasts", "enthusiastick", "enthusiastical", "enthusiasms", "enthusiastical"
        },
        "false_positives": set(),
    },

    "HEBREW REPUBLIC": {
        "forms": {
            "hebrew", "jewish", "mosaic",
            "Hebraism",
            "commonwealth of israel", "israelite commonwealth", "israelitish republic",
            "polity of the Jews"
        },
    }

}

