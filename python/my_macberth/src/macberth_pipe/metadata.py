# macberth_pipe/metadata.py
import sqlite3
from pathlib import Path
from typing import Dict

def load_doc_meta(tei_files: list[str], db_path: str) -> Dict[str, dict]:
    """
    Load metadata for a list of TEI files from the EEBO SQLite database.

    Returns:
        dict mapping doc_id ('doc0', 'doc1', ...) to metadata dict:
        {'title': ..., 'author': ..., 'year': ..., 'permalink': ...}
    """
    meta = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for i, tei_file in enumerate(tei_files):
        doc_id = f"doc{i}"
        # Extract base filename without extension
        base = Path(tei_file).stem  # e.g., "13506_1"
        # Query by permalink matching base filename
        cursor.execute(
            "SELECT title, author, year, permalink FROM eebo WHERE permalink = ?",
            (base,)
        )
        row = cursor.fetchone()
        if row:
            title, author, year, permalink = row
            meta[doc_id] = {
                "title": title or "",
                "author": author or "",
                "year": year or "",
                "permalink": permalink or ""
            }
        else:
            # fallback empty dict
            meta[doc_id] = {"title": "", "author": "", "year": "", "permalink": base}

    conn.close()
    return meta
