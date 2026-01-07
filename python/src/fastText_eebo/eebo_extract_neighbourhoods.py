#!/usr/bin/env python3
import sys
import sqlite3
import fasttext

import eebo_config as config


def load_wordlist(path):
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


queries = load_wordlist(config.QUERY_FILE)
stopwords = load_wordlist(config.STOPWORD_FILE)

print(f"[INFO] Loaded {len(queries)} query terms")
print(f"[INFO] Loaded {len(stopwords)} stopwords")

try:
    conn = sqlite3.connect(config.DB_PATH)
except Exception as e:
    print(f"[ERROR] Cannot open SQLite DB: {e}")
    sys.exit(1)

conn.execute("""
    CREATE TABLE IF NOT EXISTS neighbourhoods (
        slice_start INTEGER,
        slice_end INTEGER,
        query TEXT,
        neighbour TEXT,
        rank INTEGER,
        cosine REAL,
        PRIMARY KEY (slice_start, slice_end, query, rank)
    )
""")
conn.commit()

# Iterate over slice models
model_files = sorted(config.MODELS_DIR.glob("*.bin"))

if not model_files:
    print("[ERROR] No fastText models found")
    sys.exit(1)

for model_path in model_files:
    slice_name = model_path.stem  # e.g. 1641-1641
    start_year, end_year = map(int, slice_name.split("-"))

    print(f"[INFO] Processing slice {slice_name}")

    try:
        model = fasttext.load_model(str(model_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_path}: {e}")
        continue

    vocab = set(model.get_words())

    for query in queries:
        if query not in vocab:
            continue

        neighbors = model.get_nearest_neighbors(query, config.TOP_K * 2)

        rank = 0
        for cosine, word in neighbors:
            if word == query:
                continue
            if word in stopwords:
                continue

            rank += 1
            conn.execute("""
                INSERT OR REPLACE INTO neighbourhoods
                (slice_start, slice_end, query, neighbour, rank, cosine)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                start_year,
                end_year,
                query,
                word,
                rank,
                float(cosine)
            ))

            if rank >= config.TOP_K:
                break

    conn.commit()

conn.close()
print("[DONE] Neighbourhood extraction complete.")
