#!/usr/bin/env python3
import fasttext
import pandas as pd
from scipy.spatial.distance import cosine

import eebo_config

TARGET_WORDS = [
    "liberty", "commonwealth", "monarchy", "king", "parliament",
    "revolution", "justice", "law", "people", "tyranny"
]

slice_files = sorted(eebo_config.MODELS_DIR.glob("*.bin"))

if not slice_files:
    print(f"[ERROR] No slice models found in {eebo_config.MODELS_DIR}")
    exit(1)

print(f"[INFO] Found {len(slice_files)} slice models")

# Map: slice_name -> model
models = {}
for slice_file in slice_files:
    slice_name = slice_file.stem
    print(f"[INFO] Loading model {slice_name}")
    models[slice_name] = fasttext.load_model(str(slice_file))

# Extract word vectors
# DataFrame: rows=words, columns=slices
word_vectors = {word: {} for word in TARGET_WORDS}

for word in TARGET_WORDS:
    for slice_name, model in models.items():
        if word in model.get_words():
            vec = model.get_word_vector(word)
            word_vectors[word][slice_name] = vec
        else:
            # If word missing, store None
            word_vectors[word][slice_name] = None

# Compute cosine distance between consecutive slices
slice_names = sorted(models.keys())
drift_results = []

for word, vec_dict in word_vectors.items():
    prev_vec = None
    prev_slice = None
    for slice_name in slice_names:
        vec = vec_dict[slice_name]
        if prev_vec is not None and vec is not None:
            dist = cosine(prev_vec, vec)
        else:
            dist = None  # cannot compute drift
        drift_results.append({
            "word": word,
            "from_slice": prev_slice if prev_vec is not None else None,
            "to_slice": slice_name,
            "cosine_distance": dist
        })
        prev_vec = vec
        prev_slice = slice_name

df = pd.DataFrame(drift_results)
out_csv = eebo_config.OUT_DIR / "semantic_drift.csv"
df.to_csv(out_csv, index=False)
print(f"[DONE] Semantic drift results saved to {out_csv}")
