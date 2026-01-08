#!/usr/bin/env python
import fasttext
import sys

import eebo_config as config

# Ensure models directory exists
try:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"[ERROR] Cannot create models directory: {e}")
    sys.exit(1)


for start_year, end_year in config.SLICES:
    slice_name = f"{start_year}-{end_year}"
    slice_file = config.SLICES_DIR / f"{slice_name}.txt"
    model_file = config.MODELS_DIR / f"{slice_name}.bin"

    if not slice_file.exists():
        print(f"[WARN] Slice file missing: {slice_file}")
        continue

    print(f"[INFO] Training fastText for slice {slice_name} ({slice_file})")

    try:
        model = fasttext.train_unsupervised(
            input=str(slice_file),
            model=config.FASTTEXT_PARAMS["model"],
            dim=config.FASTTEXT_PARAMS["dim"],
            ws=config.FASTTEXT_PARAMS["ws"],
            epoch=config.FASTTEXT_PARAMS["epoch"],
            minCount=config.FASTTEXT_PARAMS["minCount"],
            minn=config.FASTTEXT_PARAMS["minn"],
            maxn=config.FASTTEXT_PARAMS["maxn"],
            thread=config.FASTTEXT_PARAMS["thread"]
        )

        model.save_model(str(model_file))
        print(f"[DONE] Saved model for slice {slice_name} to {model_file}")

    except Exception as e:
        print(f"[ERROR] Training failed for slice {slice_name}: {e}")

print("[INFO] All slices processed for fastText embeddings.")
