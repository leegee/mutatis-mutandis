# lib/stopwords_min.py
# Tokens that are not useful as independent retrieval targets
# but which remain as food for MacBERTh context windows.

STOPWORDS = {
    # Articles
    "the", "a", "an",

    # Personal / demonstrative pronouns
    "i", "me", "my", "mine",
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "thy", "thine",
    "thou", "thee", "ye",
    "who", "whom", "whose",
    "which",
    "this", "these", "those",

    # Coordinating / subordinating conjunctions
    "and", "or", "but", "nor", "yet",
    "if", "whether",
    "when", "while",
    "although", "though",
    "because", "unless",
    "than",

    # Common prepositions
    "of", "in", "to", "for", "with", "by",
    "at", "from", "as", "on", "into",
    "upon", "unto",
    "over", "under", "through",
    "after", "before", "between",
    "among", "against",
    "about", "without",
    "within", "toward", "towards",

    # Negation
    # These remain in MacBERTh's context; they simply aren't stored
    # as independent retrieval events.
    "not", "no",

    # Auxiliary / copular verbs
    "be", "is", "am", "are", "was", "were",
    "been", "being",
    "has", "have", "had",
    "do", "does", "did",

    # Early Modern English auxiliary forms
    "hast", "art", "hath", "doth", "dost",

    # Modals
    "shall", "will",
    "may", "might",
    "can", "could",
    "should", "would",
    "must",

    # Quantifiers / determiners
    "all", "any", "such", "many", "some",
    "both", "each", "few",
    "more", "most",
    "much", "less", "least",
    "several",

    # Other high-frequency function-like modifiers
    "another",
    "own",
    "same",

    # Discourse / degree adverbs with low value as independent events
    "also", "just", "only",
    "very",
}
