#!/usr/bin/env bash
set -euo pipefail

SRC="./src"
PYTHON="python"

INGEST_SCRIPT="$SRC/eebo_parse_tei.py"
MB_SENTENCE_SCRIPT="$SRC/eebo_macberth_sentence_embedding.py"
CANONICALISE_SCRIPT="$SRC/populate_canonical.py"
TRAIN_FASTTEXT_SCRIPT="$SRC/train_fastText.py"

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
echo "> Ruff"
ruff check "$SRC"

echo "> Mypy"
mypy "$SRC"

echo "> Pyright"
pyright "$SRC"

echo "All checks passed"

# Run phase
case "$PHASE" in
    ingest)
        echo "> Running ingest phase"
        "$PYTHON" "$INGEST_SCRIPT" "$@"
        ;;
    s|sentence)
        echo "> Running MacBERTh sentence phase"
        "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@"
        ;;
    t|trainft)
        echo "> Running fastText training phase"
        "$PYTHON" "$TRAIN_FASTTEXT_SCRIPT" "$@"
        ;;
    c|canonical)
        echo "> Running canonicalisation phase"
        "$PYTHON" "$CANONICALISE_SCRIPT" "$@"
        ;;
    all)
        echo "> Running full pipeline: ingest + sentence"
        "$PYTHON" "$INGEST_SCRIPT" "$@"
        "$PYTHON" "$TRAIN_FASTTEXT_SCRIPT" "$@"
        # "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@"
        "$PYTHON" "$CANONICALISE_SCRIPT" "$@"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [--phase ingest|sentence|s|trainft|t|canonical|c|all] [--limit n] [additional args...]"
        exit 1
        ;;
esac
