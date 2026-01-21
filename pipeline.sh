#!/usr/bin/env bash
set -euo pipefail

OUR_OLDPWD=`pwd`
cd python

SRC="./src" # relative to pwd set above
PYTHON="python"

# MB_SENTENCE_SCRIPT="$SRC/eebo_macberth_sentence_embedding.py"
INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
BUILD_CANONICAL_MAP="$SRC/build_canonical_map.py"
TRAIN_FASTTEXT_FOR_ORTHO_NORM="$SRC/create_fastText_ortho_norm.py"
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
    # mb|sentence) echo "# Running MacBERTh sentence phase" "$PYTHON" "$MB_SENTENCE_SCRIPT" "$@" ;;
    1|i|ingest)
    echo "# Running ingestion of TEI XML"
    "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
    ;;
    2|c|canon)
        echo "# Building canonical map"
        "$PYTHON" "$BUILD_CANONICAL_MAP" "$@"
        ;;
    3|o|ortho)
        echo "# Training fastText for orthological normalisation"
        "$PYTHON" "$TRAIN_FASTTEXT_FOR_ORTHO_NORM" "$@"
        ;;
    4|sm|spelling)
        echo "# Running spelling_map creation phase"
        "$PYTHON" "$BUILD_SPELLING_MAP_FROM_FASTTEXT" "$@"
        ;;
    5|t|slices)
        echo "# Running fastText slices training phase"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        ;;
    6|v|visual)
        echo "# Run visualisations of pre-defined keywords"
        "$PYTHON" "$VISUALISE" "$@"
        ;;
esac

cd  OUR_OLDPWD