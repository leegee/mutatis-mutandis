#!/usr/bin/env python
# eebo_extract_neighbourhoods.py
import sys
import fasttext

import eebo_config as config
import eebo_db

def load_wordlist(path):
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        sys.exit(1)
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


queries = list(CONCEPT_SETS.keys())
stopwords = load_wordlist(config.STOPWORD_FILE)

print(f"[INFO] Loaded {len(queries)} query terms")
print(f"[INFO] Loaded {len(stopwords)} stopwords")

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
            eebo_db.dbh.execute("""
                INSERT INTO neighbourhoods
                (slice_start, slice_end, query, neighbour, rank, cosine)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (slice_start, slice_end, query, rank)
                DO UPDATE SET
                    neighbour = EXCLUDED.neighbour,
                    cosine    = EXCLUDED.cosine;
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

    eebo_db.dbh.commit()

eebo_db.dbh.close()
print("[DONE] Neighbourhood extraction complete.")
