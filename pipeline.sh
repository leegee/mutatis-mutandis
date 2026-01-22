#!/usr/bin/env bash
set -euo pipefail

OUR_OLDPWD=`pwd`
cd python

SRC="./src" # relative to pwd set above
PYTHON="python"

INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
BUILD_CANONICAL_MAP="$SRC/build_canonical_map.py"
CANONICAL_FAISS_SPELLING_MAP="$SRC/canonical_faiss_spelling_map.py"
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
    3|sm|spelling)
        echo "# Creating spelling map"
        "$PYTHON" "$CANONICAL_FAISS_SPELLING_MAP" "$@"
        ;;
    4|t|slices)
        echo "# Running fastText slices training phase"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        ;;
    5|v|visual)
        echo "# Run visualisations of pre-defined keywords"
        "$PYTHON" "$VISUALISE" "$@"
        ;;
esac

cd  $OUR_OLDPWD
