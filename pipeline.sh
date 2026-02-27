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

set -- "${POSITIONAL[@]}"

pushd "$PYTHON_DIR" >/dev/null

declare -a RUN_SCRIPTS=()
declare -a RUN_ENVS=()

case "$PHASE" in
    help|-h|--help)
        echo "Please view the source...sorry"
        popd >/dev/null
        exit 0
        ;;
    1|in|ingest)
        RUN_SCRIPTS+=("$SRC/eebo_parse_tei.py")
        RUN_ENVS+=("")
        ;;
    2|train)
        # Unaligned run
        RUN_SCRIPTS+=("$SRC/slice_embedding_pipeline.py")
        RUN_ENVS+=("USE_ALIGNED_FASTTEXT_VECTORS=0")
        # Aligned run
        RUN_SCRIPTS+=("$SRC/slice_embedding_pipeline.py")
        RUN_ENVS+=("USE_ALIGNED_FASTTEXT_VECTORS=1")
        ;;
    cts|concept-timeseries)
        RUN_SCRIPTS+=("$SRC/build_concept_timeseries.py")
        RUN_ENVS+=("")
        ;;
    ps|plot-centroid-sim)
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_aligned.py")
        RUN_ENVS+=("")
        ;;
    plot-centroid-sim-knn)
        RUN_SCRIPTS+=("$SRC/vis_centroid_similarity_neighbours_aligned.py")
        RUN_ENVS+=("")
        ;;
    pca-poles)
        RUN_SCRIPTS+=("$SRC/pca_compute_eg_poles_aligned.py")
        RUN_ENVS+=("")
        ;;
    pcai|pca-poles-interactive)
        RUN_SCRIPTS+=("$SRC/pca_interactive_liberty_plot.py")
        RUN_ENVS+=("")
        ;;
    umap-liberty)
        RUN_SCRIPTS+=("$SRC/umap_interactive_liberty_umap.py")
        RUN_ENVS+=("")
        ;;
    exp|concept-explorer)
        RUN_SCRIPTS+=("$SRC/concept_neighbour_explorer.py" "$SRC/concept_neighbour_explorer_plot.py")
        RUN_ENVS+=("" "")
        ;;
    uc|usage-cluster)
        RUN_SCRIPTS+=("$SRC/usage_clusterer2.py")
        RUN_ENVS+=("")
        ;;
    ucv|usage-cluster-viz)
        RUN_SCRIPTS+=("$SRC/viz_usage_clusters_interactive.py")
        RUN_ENVS+=("")
        ;;
    ucs|usage-cluster-sankey)
        RUN_SCRIPTS+=("$SRC/viz_usage_clusters_sankey.py")
        RUN_ENVS+=("")
        ;;
    sa|secularisation_analysis)
        RUN_SCRIPTS+=("$SRC/secularisation_analysis.py")
        RUN_ENVS+=("")
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

# Toolchain must exist
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

# Execute scripts with optional per-script environment
for i in "${!RUN_SCRIPTS[@]}"; do
    script="${RUN_SCRIPTS[i]}"
    env_prefix="${RUN_ENVS[i]}"
    echo "# Running $script with $env_prefix"
    if [[ -n "$env_prefix" ]]; then
        env $env_prefix "$PYTHON" "$script" "$@"
    else
        "$PYTHON" "$script" "$@"
    fi
done

popd >/dev/null