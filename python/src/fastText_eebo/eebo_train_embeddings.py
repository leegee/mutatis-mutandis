#!/usr/bin/env python3
import fasttext
import sys

import eebo_config

# Ensure models directory exists
try:
    eebo_config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Cannot create models directory: {e}")
    sys.exit(1)

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

FASTTEXT_PARAMS = {
    "model": "skipgram",      # Skip-gram model
    "dim": 200,               # Word vector dimensionality
    "ws": 5,                  # Context window size
    "epoch": 10,              # Number of epochs
    "minCount": 1,            # Keep all words
    "thread": 4,              # Adjust to your CPU
    "minn": 3,                # Subword ngram min length
    "maxn": 6,                # Subword ngram max length
}

for start_year, end_year in slices:
    slice_name = f"{start_year}-{end_year}"
    slice_file = eebo_config.SLICES_DIR / f"{slice_name}.txt"
    model_file = eebo_config.MODELS_DIR / f"{slice_name}.bin"

    if not slice_file.exists():
        print(f"[WARN] Slice file missing: {slice_file}")
        continue

    print(f"[INFO] Training fastText for slice {slice_name} ({slice_file})")

    try:
        model = fasttext.train_unsupervised(
            input=str(slice_file),
            model=FASTTEXT_PARAMS["model"],
            dim=FASTTEXT_PARAMS["dim"],
            ws=FASTTEXT_PARAMS["ws"],
            epoch=FASTTEXT_PARAMS["epoch"],
            minCount=FASTTEXT_PARAMS["minCount"],
            minn=FASTTEXT_PARAMS["minn"],
            maxn=FASTTEXT_PARAMS["maxn"],
            thread=FASTTEXT_PARAMS["thread"]
        )

        model.save_model(str(model_file))
        print(f"[DONE] Saved model for slice {slice_name} to {model_file}")

    except Exception as e:
        print(f"[ERROR] Training failed for slice {slice_name}: {e}")

print("[INFO] All slices processed for fastText embeddings.")
