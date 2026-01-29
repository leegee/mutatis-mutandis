#!/usr/bin/env bash
set -euo pipefail

SRC="./src"
PYTHON="python"
PHASE="help"
OUR_OLDPWD=$(pwd)

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

# Use an array to hold scripts for the selected phase
RUN_SCRIPTS=()

case "$PHASE" in
    1|in|ingest)
        echo "# Running corpus prep and ingestion"
        RUN_SCRIPTS+=("$SRC/eebo_parse_tei.py")
        ;;
    2|train)
        echo "# Phase 2: Create training files and train fastText"
        RUN_SCRIPTS+=("$SRC/generate_training_files.py")
        RUN_SCRIPTS+=("$SRC/train_slice_fasttext.py")
        ;;
    2a|training-files)
        RUN_SCRIPTS+=("$SRC/generate_training_files.py")
        ;;
    2b|train-fasttext)
        RUN_SCRIPTS+=("$SRC/train_slice_fasttext.py")
        ;;
    3|f|faiss)
        RUN_SCRIPTS+=("$SRC/build_faiss_slice_indexes.py")
        ;;
    4|v|token-vectors)
        RUN_SCRIPTS+=("$SRC/generate_token_embeddings.py")
        ;;
    5|c|concept-timeseries)
        RUN_SCRIPTS+=("$SRC/build_concept_timeseries.py")
        ;;
    6|p|plot)
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity.py")
        ;;
    7|n|knn)
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_neighbours.py")
        ;;
    8a|p|pca)
        RUN_SCRIPTS+=("$SRC/pca_compute_eg_poles.py")
        ;;
    8b|pcai|pcai)
        RUN_SCRIPTS+=("$SRC/pca_interactive_liberty_plot.py")
        ;;
    8c|u|umap)
        RUN_SCRIPTS+=("$SRC/umap_interactive_liberty_umap.py")
        ;;
    9a|k|knn)
        RUN_SCRIPTS+=("$SRC/knn_audit.py")
        ;;
    9b|html)
        RUN_SCRIPTS+=("$SRC/knn_audit_html.py")
        ;;
    *)
        echo "! No phase selected or invalid phase: $PHASE"
        cd "$OUR_OLDPWD"
        exit 1
        ;;
esac

if [[ ${#RUN_SCRIPTS[@]} -eq 0 ]]; then
    echo "! No scripts to run for phase: $PHASE"
    cd "$OUR_OLDPWD"
    exit 1
fi

echo "# Running Ruff"
ruff check "$SRC"

echo "# Running Mypy"
mypy "$SRC"

echo "# Running Pyright"
pyright "$SRC"

echo "# All checks passed"

# Execute each script in sequence
for script in "${RUN_SCRIPTS[@]}"; do
    echo "# Running $script"
    "$PYTHON" "$script" "$@"
done

cd "$OUR_OLDPWD"
