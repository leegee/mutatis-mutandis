from cannonical_rules import CanonicalRule, CanonicalRules

CORPUS_MIN_YEAR = 0
CORPUS_MAX_YEAR = 2026
FILTER_DOCUMENT_SIZE = False
MIN_TOKENS_IN_DOC = 200
MAX_TOKENS_IN_DOC = 400_000

CONCEPT_SETS: CanonicalRules = {
    "WHITE": {
        "forms": {
            "white",
            "whit",
            "hwit",
            "whyte",
            "whight",
            "whyt", "whyte", "hwyte",
            "whitt",
            "auburn" # early 15c off-white cf alb, but meaning shifted 16c. to "reddish-brown" under influence of Middle English brun
        },
        "false_positives": {
            "wight",
            "whiten",
        },
    },
}


# In lib.corpus_config (or here for a quick prototype)
PHRASE_SETS: CanonicalRules = {
    "white_as_wool": {
        "forms": [
            # surface patterns; can be expanded later with regex / lemmas
            ("hair", "as", "white", "as", "wool"),
            ("white", "as", "wool"),
            ("white", "like", "wool"),
            ("hore", "as", "wool"),          # early spelling
            # …
        ],
        "false_positives": [],
    },
    # other formulaic phrases …
}
