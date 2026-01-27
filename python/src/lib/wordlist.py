# src/lib/wordlist.py

from typing import Set
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
