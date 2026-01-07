#!/usr/bin/env python3
import sqlite3
import sys

import eebo_config

# Ensure slices directory exists
try:
    eebo_config.SLICES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Cannot create slices directory: {e}")
    sys.exit(1)


# Define slices' year ranges
slices = [
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
    (1661, 1665),
]


try:
    conn = sqlite3.connect(eebo_config.DB_PATH)
except Exception as e:
    print(f"[ERROR] Cannot open SQLite DB at {eebo_config.DB_PATH}: {e}")
    sys.exit(1)

cursor = conn.cursor()


# Process each slice
for start_year, end_year in slices:
    slice_name = f"{start_year}-{end_year}"
    out_path = eebo_config.SLICES_DIR / f"{slice_name}.txt"
    print(f"[INFO] Processing slice {slice_name}")

    cursor.execute(
        "SELECT doc_id FROM documents WHERE date BETWEEN ? AND ?",
        (start_year, end_year)
    )
    doc_ids = [row[0] for row in cursor.fetchall()]
    print(f"[INFO] {len(doc_ids)} documents in slice {slice_name}")

    # Concatenate text files
    texts = []
    missing_count = 0
    for doc_id in doc_ids:
        text_path = eebo_config.PLAIN_DIR / f"{doc_id}.txt"
        if text_path.exists():
            texts.append(text_path.read_text(encoding="utf-8"))
        else:
            missing_count += 1
            print(f"[WARN] Missing text file for {doc_id}")

    if missing_count:
        print(f"[WARN] {missing_count} missing files in slice {slice_name}")

    # Write concatenated slice
    try:
        out_path.write_text("\n".join(texts), encoding="utf-8")
        print(f"[DONE] Wrote slice file: {out_path} ({len(texts)} texts)")
    except Exception as e:
        print(f"[ERROR] Failed to write slice {slice_name}: {e}")

conn.close()
print("[INFO] All slices processed.")
