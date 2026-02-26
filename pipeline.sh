#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
SRC="$PYTHON_DIR/src"

PYTHON="${PYTHON:-python}"

PHASE="help"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase|-p)
            [[ $# -ge 2 ]] || { echo "Missing value for $1"; exit 1; }
            PHASE="$2"
            shift 2
            ;;
        --phase=*|-p=*)
            PHASE="${1#*=}"
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

# Restore positional args
set -- "${POSITIONAL[@]}"

pushd "$PYTHON_DIR" >/dev/null

RUN_SCRIPTS=()

case "$PHASE" in
    help|-h|--help)
        echo "Please view the source...sorry"
        popd >/dev/null
        exit 0
        ;;
    1|in|ingest)
        echo "# Running corpus prep and ingestion"
        RUN_SCRIPTS+=("$SRC/eebo_parse_tei.py")
        ;;
    2|train)
        echo "# Phase 2: Training pipeline"
        RUN_SCRIPTS+=("$SRC/generate_training_files.py")
        RUN_SCRIPTS+=("$SRC/train_slice_fasttext.py")
        RUN_SCRIPTS+=("$SRC/align.py")
        ;;
    2a|training-files)
        RUN_SCRIPTS+=("$SRC/generate_training_files.py")
        ;;
    2b|train-fasttext)
        RUN_SCRIPTS+=("$SRC/train_slice_fasttext.py")
        ;;
    2c|align)
        RUN_SCRIPTS+=("$SRC/align.py")
        ;;
    3|f|faiss)
        RUN_SCRIPTS+=("$SRC/build_faiss_slice_indexes.py")
        ;;
    4|v|token-vectors)
        RUN_SCRIPTS+=("$SRC/generate_token_embeddings.py")
        ;;
    concept-timeseries)
        RUN_SCRIPTS+=("$SRC/build_concept_timeseries.py")
        ;;
    plot-centroid-sim)
        # RUN_SCRIPTS+=("$SRC/vis_centroid_similarity.py")
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_aligned.py")
        ;;
    plot-centroid-sim-knn)
        # RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_neighbours.py")
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_neighbours_aligned.py")
        ;;
    pca-poles)
        # RUN_SCRIPTS+=("$SRC/pca_compute_eg_poles.py")
        RUN_SCRIPTS+=("$SRC/pca_compute_eg_poles_aligned.py")
        ;;
    pca-poles-interactive)
        RUN_SCRIPTS+=("$SRC/pca_interactive_liberty_plot.py")
        ;;
    umap-liberty)
        RUN_SCRIPTS+=("$SRC/umap_interactive_liberty_umap.py")
        ;;
    exp)
        RUN_SCRIPTS+=("$SRC/concept_neighbour_explorer.py")
        ;;
    expp|exp-plot)
        RUN_SCRIPTS+=("$SRC/concept_neighbour_explorer_plot.py")
        ;;
    uc|usage-cluster)
        RUN_SCRIPTS+=("$SRC/usage_clusterer2.py")
        ;;
    ucv|usage-cluster-viz)
        RUN_SCRIPTS+=("$SRC/viz_usage_clusters_interactive.py")
        ;;
    ucs|usage-cluster-sankey)
        RUN_SCRIPTS+=("$SRC/viz_usage_clusters_sankey.py")
        ;;
    *)
        echo "! Invalid phase: $PHASE"
        popd >/dev/null
        exit 1
        ;;
esac

# Ensure scripts exist
for script in "${RUN_SCRIPTS[@]}"; do
    [[ -f "$script" ]] || { echo "Script not found: $script"; popd >/dev/null; exit 1; }
done

# Toolchain must exist (fail if missing)
command -v ruff >/dev/null || { echo "ruff not installed"; popd >/dev/null; exit 1; }
command -v mypy >/dev/null || { echo "mypy not installed"; popd >/dev/null; exit 1; }
command -v pyright >/dev/null || { echo "pyright not installed"; popd >/dev/null; exit 1; }

echo "# Running Ruff"
ruff check "$SRC"

echo "# Running Mypy"
mypy "$SRC"

echo "# Running Pyright"
pyright "$SRC"

echo "# All checks passed"

# Execute scripts
for script in "${RUN_SCRIPTS[@]}"; do
    echo "# Running $script"
    "$PYTHON" "$script" "$@"
done

popd >/dev/null
