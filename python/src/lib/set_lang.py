#!/usr/bin/env python

"""
lib/set_lang.py -

Parse the documents table and label the document as Latin if common Latin words are
detected and common English words are not.

TODO Make sure this is called prior to materialising pamphlet_corpus view.
"""

from psycopg.rows import dict_row
from lib.eebo_db import get_connection
from lib.eebo_logging import logger


LATIN_FUNCTION = {
    'et','non','est','aut','vel','quod','cum','per','ad','in','de','ex','ut','sed','nec','neque'
}

ENGLISH_FUNCTION = {
    'the','and','of','to','in','that','is','it','for','with','as','on','be','by','this','which'
}


def set_document_languages() -> None:
    conn = get_connection(application_name="lang_detect")

    logger.info("Starting document language detection")

    with conn.transaction():
        with conn.cursor(row_factory=dict_row) as cur:

            logger.info("Computing language signals per document")

            cur.execute("""
                SELECT
                    d.doc_id,

                    COUNT(*) FILTER (
                        WHERE lower(t.token) = ANY(%s)
                           OR right(lower(t.token), 2) IN ('us','um','ae','is')
                           OR right(lower(t.token), 4) IN ('orum','arum')
                    ) AS latin_tokens,

                    COUNT(*) FILTER (
                        WHERE lower(t.token) = ANY(%s)
                    ) AS english_tokens,

                    COUNT(*) AS total_tokens

                FROM documents d
                JOIN tokens t ON t.doc_id = d.doc_id
                GROUP BY d.doc_id
                HAVING COUNT(*) > 500
            """, (list(LATIN_FUNCTION), list(ENGLISH_FUNCTION)))

            updates = []

            for row in cur:
                doc_id = row["doc_id"]
                latin = row["latin_tokens"] or 0
                english = row["english_tokens"] or 0
                total = row["total_tokens"]

                latin_ratio = latin / total
                english_ratio = english / total
                latin_score = latin_ratio - english_ratio

                # RULE:
                # Pure Latin texts show:
                # latin_ratio around 0.30–0.40
                # english_ratio around 0.00
                if latin_score > 0.15 and english_ratio < 0.03:
                    lang = "lat"
                else:
                    lang = "eng"

                updates.append((lang, doc_id))

            logger.info(f"Updating {len(updates)} documents")

            cur.executemany(
                "UPDATE documents SET lang = %s WHERE doc_id = %s",
                updates
            )

    logger.info("Language detection complete")


if __name__ == "__main__":
    set_document_languages()
