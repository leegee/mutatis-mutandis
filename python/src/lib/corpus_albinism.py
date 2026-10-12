from lib.corpus_db import get_connection
from lib.cannonical_rules import CanonicalRule, CanonicalRules
from lib.corpus_logging import logger

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


CORPUS_SQL = """
WITH
white_tokens AS (
    SELECT corpus, doc_id, token_idx
    FROM tokens
    WHERE lower(token) IN (
        'white', 'whyte', 'whight', 'whit',
        'whyt', 'wight', 'wite', 'whiet'
    )
),
appearance_tokens AS (
    SELECT corpus, doc_id, token_idx
    FROM tokens
    WHERE lower(token) IN (
        'hair', 'haire', 'hayre', 'hayr', 'heare',
        'hoary', 'hoarie',
        'skin', 'skinne', 'skyn', 'complexion', 'flesh',
        'fur', 'furre', 'furr', 'coat', 'fleece',
        'pale', 'pallid', 'pallide', 'wan', 'wann',
        'wanne', 'bloodless',
        'luminous', 'luminouse', 'luminus', 'luminious',
        'glowing', 'glow', 'glowe', 'glowinge', 'glowed',
        'shining', 'shyning', 'bright', 'brightness',
        'radiant', 'radiance', 'resplendent', 'resplendence',
        'glistering', 'glisteringe', 'gleaming', 'gleaminge',
        'gleam', 'gleame', 'glittering', 'glitteringe',
        'lucent', 'lucid'
    )
),
light_tokens AS (
    SELECT corpus, doc_id, token_idx
    FROM tokens
    WHERE lower(token) IN (
        'luminous', 'luminouse', 'luminus', 'luminious',
        'glowing', 'glow', 'glowe', 'glowinge', 'glowed',
        'shining', 'shyning', 'bright', 'brightness',
        'radiant', 'radiance', 'resplendent', 'resplendence',
        'glistering', 'glisteringe', 'gleaming', 'gleaminge',
        'gleam', 'gleame', 'glittering', 'glitteringe',
        'lucent', 'lucid'
    )
),
supernatural_tokens AS (
    SELECT corpus, doc_id, token_idx
    FROM tokens
    WHERE lower(token) IN (
        'ghost', 'ghosts', 'ghostly',
        'phantom', 'phantoms', 'phantasme',
        'apparition', 'apparitions',
        'spirit', 'spirits', 'sprite', 'sprites',
        'elf', 'elves', 'elves',
        'fairy', 'fairies', 'faerie', 'faeries', 'fay', 'fays',
        'spectre', 'spectres', 'specter', 'specters',
        'wraith', 'wraiths', 'angel', 'angels',
        'demon', 'demons', 'daemon', 'daemons',
        'goblin', 'goblins', 'hobgoblin', 'hobgoblins',
        'incubus', 'succubus'
    )
),
selected_docs AS (
    -- Route 1: whiteness near appearance or luminosity
    SELECT w.corpus, w.doc_id
    FROM white_tokens w
    JOIN appearance_tokens a
      ON a.corpus = w.corpus
     AND a.doc_id = w.doc_id
     AND a.token_idx BETWEEN w.token_idx - 10 AND w.token_idx + 10

    UNION

    -- Route 2: luminosity near a supernatural figure
    SELECT l.corpus, l.doc_id
    FROM light_tokens l
    JOIN supernatural_tokens s
      ON s.corpus = l.corpus
     AND s.doc_id = l.doc_id
     AND s.token_idx BETWEEN l.token_idx - 10 AND l.token_idx + 10

    UNION

    -- Route 3: explicit whiteness near a supernatural figure
    SELECT w.corpus, w.doc_id
    FROM white_tokens w
    JOIN supernatural_tokens s
      ON s.corpus = w.corpus
     AND s.doc_id = w.doc_id
     AND s.token_idx BETWEEN w.token_idx - 10 AND w.token_idx + 10
)
SELECT
    d.corpus,
    d.doc_id,
    d.pub_year,
    d.author,
    d.title,
    d.filepath
FROM selected_docs s
JOIN documents d ON d.doc_id = s.doc_id
ORDER BY d.pub_year NULLS LAST, d.title
"""


def select_white_corpus_doc_ids():
    con = get_connection()
    try:
        with con.cursor() as cur:
            cur.execute(CORPUS_SQL)
            rows = cur.fetchall()

        logger.info("White corpus selection returned %d documents", len(rows))

        for row in rows:
            logger.info(
                "Selected document: corpus=%s doc_id=%s year=%s author=%s title=%s",
                row[0], row[1], row[2], row[3], row[4]
            )

        return [row[1] for row in rows]

    finally:
        con.close()

if __name__ == "__main__":
    select_white_corpus_doc_ids()
