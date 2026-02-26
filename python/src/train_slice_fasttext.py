#!/usr/bin/env python
"""
train_slice_fasttext.py

Train slice-specific fastText models on the plain text files generated
by generate_training_files.py.
"""

from typing import Tuple
from pathlib import Path
import fasttext

from lib.eebo_logging import logger
from lib.eebo_config import SLICES, SLICES_DIR, FASTTEXT_SLICE_MODEL_DIR, FASTTEXT_PARAMS

FASTTEXT_SLICE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

def slice_model_path(slice_range: Tuple[int, int]) -> Path:
    """
    Return the fastText model path for a given slice.
    slice_range: (start_year, end_year)
    """
    start, end = slice_range
    return FASTTEXT_SLICE_MODEL_DIR / f"slice_{start}_{end}.bin"


def train_slice(slice_file: Path, start: int, end: int) -> Path:
    """
    Train fastText skip-gram model on a single slice file.
    Saves the model as slice_{start}_{end}.bin in FASTTEXT_SLICE_MODEL_DIR
    """
    logger.info(f"Training slice {start}-{end} on {slice_file} ...")

    model = fasttext.train_unsupervised(
        input=str(slice_file),
        **FASTTEXT_PARAMS
    )

    model_path = slice_model_path((start, end))
    model.save_model(str(model_path))
    logger.info(f"Saved model for slice {start}-{end} → {model_path}")
    return model_path


def main():
    for start, end in SLICES:
        slice_file = SLICES_DIR / f"{start}-{end}.txt"
        if not slice_file.exists():
            logger.info(f"Warning: slice file {slice_file} missing, skipping")
            continue
        train_slice(slice_file, start, end)

    logger.info(f"All slice models saved to {FASTTEXT_SLICE_MODEL_DIR}")


if __name__ == "__main__":
    main()
