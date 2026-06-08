# src/lib/wordlist.py

from typing import Set
import nltk
from nltk.corpus import stopwords

import lib.eebo_config as config

def load_wordlist() -> Set[str]:
    """
    Loads a wordlist from the file specified in config.STOPWORD_FILE.
    Returns a set of lowercase words for fast lookup.
    """
    wordlist: Set[str] = set()

    try:
        with open(config.STOPWORD_FILE, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    wordlist.add(word.lower())
    except FileNotFoundError as err:
        raise RuntimeError(f"Stopword file not found: {config.STOPWORD_FILE}") from err


    return wordlist

nltk.download("stopwords")
STOPWORDS: Set[str] = set(stopwords.words("english"))
STOPWORDS.update(load_wordlist())

EEBO_EXTRA = {
    "s", "thou", "thee", "thy", "thine", "hath", "doth", "art", "ye", "v",
    "may", "shall", "upon", "us", "yet", "would", "one", "unto", "said",
    "de", "c", "also", "do", "day", "bee", "be", "doe", "therefore"
}
STOPWORDS.update(EEBO_EXTRA)

