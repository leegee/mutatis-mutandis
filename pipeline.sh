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

RUN=""

# Run phase
case "$PHASE" in
    1|in|ingest)
        echo "# Running corpus prep and ingestion"
        RUN="$SRC/eebo_parse_tei.py"
        ;;
    2a|training-files)
        echo "# Create files of slices for training fastText"
        RUN="$SRC/generate_training_files.py"
        ;;
    2b|train-fasttext)
        echo "# Training fastText on slices to create semantic space"
        RUN="$SRC/train_slice_fasttext.py"
        ;;
    3|f|faiss)
        echo "# Create FAISS of fastText"
        RUN="$SRC/build_faiss_slice_indexes.py"
        ;;
    4|v|token-vectors)
        echo "# Materialise vectors/embeddings from token vectors"
        RUN="$SRC/generate_token_embeddings.py"
        ;;
    5|c|concept-timeseries)
        echo "# Generate concept model stats"
        RUN="$SRC/build_concept_timeseries.py"
        ;;
    6|p|plot)
        echo "# Plot centroid similiarity"
        RUN="$SRC/vis_centroid_similarity.py"
        ;;
    7|n|knn)
        echo "# Plot centroid nearest neighbours"
        RUN="$SRC/vis_centroid_similarity_neighbours.py"
        ;;
    8a|p|pca)
        echo "# Compute PCA conceptual poles"
        RUN="$SRC/pca_compute_eg_poles.py"
        ;;
    8b|pcai|pcai)
        echo "# Interactive plot of PCA Liberty"
        RUN="$SRC/pca_interactive_liberty_plot.py"
        ;;
    8c|u|umap)
        echo "# Interactive UMAP plot of Liberty"
        RUN="$SRC/umap_interactive_liberty_umap.py"
        ;;
esac

if [[ -z "$RUN" ]]; then
    echo "! No phase selected or invalid phase: $PHASE"
    exit 1
fi

echo "# Shall run $RUN"

echo "# Running Ruff"
ruff check "$SRC"

echo "# Running Mypy"
mypy "$SRC"

echo "# Running Pyright"
pyright "$SRC"

echo "# All checks passed"

echo "# Running $RUN"
"$PYTHON" "$RUN" "$@"

cd "$OUR_OLDPWD"
