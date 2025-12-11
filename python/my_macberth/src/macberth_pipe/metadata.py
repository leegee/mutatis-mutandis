import sqlite3
from pathlib import Path
from typing import Dict
import logging

def load_doc_meta(tei_files: list[str], db_path: str) -> Dict[str, dict]:
    """
    Load metadata for TEI files using the EEBO SQLite database.

    Returns:
        dict mapping doc_id ('doc0', etc.) to metadata:
        {'title': ..., 'author': ..., 'year': ..., 'permalink': ...}
    """
    meta = {}

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i, tei_file in enumerate(tei_files):
        doc_id = f"doc{i}"

        # File stem "13506_1" must be mapped to DB format "13506 1"
        base_id = Path(tei_file).stem.replace("_", " ")

        cursor.execute(
            """
            SELECT title, author, year, permalink
            FROM eebo
            WHERE philo_div1_id = ?
            """,
            (base_id,)
        )

        row = cursor.fetchone()

        if row:
            title, author, year, permalink = row
            meta[doc_id] = {
                "title": title or "",
                "author": author or "",
                "year": year or "",
                "permalink": permalink or "",
            }
            # logging.debug("Metadata found for %s: %s", base_id, meta[doc_id])
        else:
            logging.warning("No metadata found for '%s' in DB", base_id)
            meta[doc_id] = {
                "title": "N/A",
                "author": "N/A",
                "year": "N/A",
                "permalink": base_id,  # fallback
            }

    conn.close()
    return meta
