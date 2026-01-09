#!/usr/bin/env python
import sqlite3
import re
import sys

import eebo_config


def update_document_dates():
    try:
        conn = sqlite3.connect(eebo_config.DB_PATH)
    except Exception as e:
        print(f"[ERROR] Cannot open SQLite DB at {eebo_config.DB_PATH}: {e}")
        sys.exit(1)

    cursor = conn.cursor()

    # Add 'date' column if it does not exist
    cursor.execute("PRAGMA table_info(documents)")
    columns = [col[1] for col in cursor.fetchall()]
    if "date" not in columns:
        print("[INFO] Adding 'date' column to documents table")
        cursor.execute("ALTER TABLE documents ADD COLUMN date INTEGER")
        conn.commit()
    else:
        print("[INFO] 'date' column already exists")

    # Extract first 4-digit year from source_date_raw
    cursor.execute("SELECT doc_id, source_date_raw FROM documents")
    rows = cursor.fetchall()

    update_count = 0
    for doc_id, source_date_raw in rows:
        if not source_date_raw:
            continue
        match = re.search(r'\b(1[5-7][0-9]{2})\b', source_date_raw)
        if match:
            year = int(match.group(1))
            cursor.execute(
                "UPDATE documents SET date = ? WHERE doc_id = ?",
                (year, doc_id)
            )
            update_count += 1

    conn.commit()
    print(f"[INFO] Updated 'date' column for {update_count} documents")

    cursor.execute(
        "SELECT MIN(date), MAX(date) FROM documents WHERE date IS NOT NULL"
    )
    min_year, max_year = cursor.fetchone()
    print(f"[INFO] Document date range: {min_year} – {max_year}")

    conn.close()


if __name__ == "__main__":
    update_document_dates()
