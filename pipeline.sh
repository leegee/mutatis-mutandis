#!/usr/bin/env bash
set -euo pipefail

SRC="./src" # relative to pwd set above
PYTHON="python"
PHASE="help"
OUR_OLDPWD=`pwd`

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

cd python

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
        RUN="$SRC/eebo_parse_tei.py"
        ;;
    2|f|training-files)
        echo "# Create files of slices for training fastText"
        RUN="$SRC/generate_training_files.py"
        ;;
    3|t|train-fasttext)
        echo "# Training fastText on slices"
        RUN="$SRC/train_slice_fasttext.py"
        ;;
    4|v|gen-token-vectors)
        echo "# Generate embeddings: token vectors"
        RUN="$SRC/generate_token_embeddings.py"
        ;;

esac

echo "# Running $RUN"
"$PYTHON" "$RUN" "$@"

cd "$OUR_OLDPWD"
