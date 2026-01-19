#!/usr/bin/env bash
set -euo pipefail

SRC="./src"
PYTHON="python"

INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
# MB_SENTENCE_SCRIPT="$SRC/eebo_macberth_sentence_embedding.py"
TRAIN_FASTTEXT_FOR_NORMALISATION="$SRC/create_fastText_ortho_canon.py"
MAKE_FASTTEXT_SLICES="$SRC/train_fastText_canonicalised_slices.py"
BUILD_CANONICAL_MAP="$SRC/build_canonical_map.py"

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
            POSITIONAL+=("$1") # collect extra args for Python scripts
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

# Run phase
case "$PHASE" in
    i|ingest)
        echo "# Running ingest phase"
        "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        ;;
    # s|sentence)
    #     echo "# Running MacBERTh sentence phase"
    #     "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@"
    #     ;;
    t|trainft)
        echo "# Running fastText training phase"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        ;;
    c|canoical)
        echo "# Running caonicalisation phase"
        "$PYTHON" "$TRAIN_FASTTEXT_FOR_NORMALISATION" "$@"
        ;;
    all)
        echo "# Running full pipeline: ingest + sentence"
        "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        # "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@"
        "$PYTHON" "$TRAIN_FASTTEXT_FOR_NORMALISATION" "$@"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [--phase ingest|i|sentence|s|trainft|t|canonical|c|all] [--limit n] [additional args...]"
        exit 1
        ;;
esac
