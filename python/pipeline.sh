#!/usr/bin/env bash
set -euo pipefail

SRC="./src"
PYTHON="python"

# MB_SENTENCE_SCRIPT="$SRC/eebo_macberth_sentence_embedding.py"
INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
BUILD_CANONICAL_MAP="$SRC/build_canonical_map.py"
TRAIN_FASTTEXT_FOR_NORMALISATION="$SRC/create_fastText_ortho_canon.py"
BUILD_SPELLING_MAP_FROM_FASTTEXT="$SRC/build_spelling_map_from_fastText.py"
MAKE_FASTTEXT_SLICES="$SRC/train_fastText_canonicalised_slices.py"
VISUALISE="$SRC/semantic_drift_analysis.py"

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
    # s|sentence) echo "# Running MacBERTh sentence phase" "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@" ;;
    i|ingest)
        echo "# Running ingestion of TEI XML"
        "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        ;;
    t|trainft)
        echo "# Running fastText training phase"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        ;;
    cm|canon)
        echo "# Building canonical map"
        "$PYTHON" "$BUILD_CANONICAL_MAP" "$@"
        ;;
    sm|spelling_map)
        echo "# Running spelling_map creation phase"
        "$PYTHON" "$BUILD_SPELLING_MAP_FROM_FASTTEXT" "$@"
        ;;

    c|canoical)
        echo "# Training fastText for normalisation"
        "$PYTHON" "$TRAIN_FASTTEXT_FOR_NORMALISATION" "$@"
        ;;
    v|visual)
        echo "# Run visualisations of pre-defined keywords"
        "$PYTHON" "$VISUALISE" "$@"
        ;;
    all)
        echo "# Running full pipeline: ingest + sentence"
        # "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        # # "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@"
        # "$PYTHON" "$TRAIN_FASTTEXT_FOR_NORMALISATION" "$@"
        INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
        BUILD_CANONICAL_MAP="$SRC/build_canonical_map.py"
        TRAIN_FASTTEXT_FOR_NORMALISATION="$SRC/create_fastText_ortho_canon.py"
        BUILD_SPELLING_MAP_FROM_FASTTEXT="$SRC/build_spelling_map_from_fastText.py"
        MAKE_FASTTEXT_SLICES="$SRC/train_fastText_canonicalised_slices.py"
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: $0 [--phase ingest|i|sentence|s|trainft|t|canonical|c|all] [--limit n] [additional args...]"
        exit 1
        ;;
esac
