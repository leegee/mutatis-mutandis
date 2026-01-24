#!/usr/bin/env bash
set -euo pipefail

OUR_OLDPWD=`pwd`
cd python

SRC="./src" # relative to pwd set above
PYTHON="python"

INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
CREATE_TRAINING_FILES="$SRC/generate_training_files.py"
TRAIN_SLICE_FASTTEXT="$SRC/train_slice_fasttext.py"

# Default values
PHASE="all"

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --phase=*)
            PHASE="${key#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Restore positional arguments
set -- "${POSITIONAL[@]}"

# Run checks
echo "# Running Ruff"
ruff check "$SRC"

echo "# Running Mypy"
mypy "$SRC"

echo "# Running Pyright"
pyright "$SRC"

echo "# All checks passed"

RUN=""

# Run phase
case "$PHASE" in
    1|i|ingest)
        echo "# Running ingestion of TEI XML"
        RUN="$INIT_AND_INGEST_XML"
        ;;
    2|f|training-files)
        echo "# Create training slices"
        RUN="$CREATE_TRAINING_FILES"
        ;;
    3|t|train)
        echo "# Training fastText on slices"
        RUN="$TRAIN_SLICE_FASTTEXT"
        ;;
esac

echo "# Running $RUN"
"$PYTHON" "$RUN" "$@"

cd "$OUR_OLDPWD"
