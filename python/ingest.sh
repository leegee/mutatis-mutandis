#!/usr/bin/env bash
set -euo pipefail

SRC="./src"
PYTHON="python"

INGEST_SCRIPT="$SRC/eebo_parse_tei.py"
SENTENCE_SCRIPT="$SRC/eebo_macberth_sentence_embedding.py"

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
        echo "> Running sentence phase"
        "$PYTHON" "$SENTENCE_SCRIPT" "$@"
        ;;
    all)
        echo "> Running full pipeline: ingest + sentence"
        "$PYTHON" "$INGEST_SCRIPT" "$@"
        "$PYTHON" "$SENTENCE_SCRIPT" "$@"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [--phase ingest|sentence|s|all] [additional args...]"
        exit 1
        ;;
esac
