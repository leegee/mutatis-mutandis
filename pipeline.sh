#!/usr/bin/env bash
set -euo pipefail

OUR_OLDPWD=`pwd`
cd python

SRC="./src" # relative to pwd set above
PYTHON="python"

INIT_AND_INGEST_XML="$SRC/eebo_parse_tei.py"
BUILD_SPELLNG_MAP="$SRC/build_spellng_map.py"
REFINE_SPELLING_MAP="$SRC/refine_spelling_map.py"
MAKE_FASTTEXT_SLICES="$SRC/train_fastText_canonicalised_slices.py"

EXTRACT_NEIGHBOURHOODS="$SRC/extract_neighbourhoods.py"

# Fanning out on neighbours for conceptual meta-neighbourhood
ANNOTATE_NEIGHBOR_ROLES="$SRC/annotate_neighbors_with_roles.py"
BUILD_ROLE_PROFILES="$SRC/build_role_profiles_per_slice.py"

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
            POSITIONAL+=("$1")
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
    1|i|ingest)
        echo "# Running ingestion of TEI XML"
        "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        ;;
    2|c|canon)
        echo "# Building canonical map"
        "$PYTHON" "$BUILD_SPELLNG_MAP" "$@"
        ;;
    3|sm|spelling)
        echo "# Creating spelling map"
        "$PYTHON" "$REFINE_SPELLING_MAP" "$@"
        ;;
    4|t|slices)
        echo "# Running fastText slices training phase"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        ;;
    5|n|Neighbours)
        echo "# Neighbourhood extraction per slice"
        "$PYTHON" "$EXTRACT_NEIGHBOURHOODS" "$@"
        ;;
    6|r|roles)
        echo "# Annotating FAISS neighbours with conceptual roles"
        "$PYTHON" "$ANNOTATE_NEIGHBOR_ROLES" "$@"
        ;;
    7|p|profiles)
        echo "# Building role-conditioned neighbour profiles per slice"
        "$PYTHON" "$BUILD_ROLE_PROFILES" "$@"
        ;;
    8|v|visual)
        echo "# Run visualisations of pre-defined keywords"
        "$PYTHON" "$VISUALISE" "$@"
        ;;
    all)
        echo "# Running full pipeline"
        "$PYTHON" "$INIT_AND_INGEST_XML" "$@"
        "$PYTHON" "$BUILD_SPELLNG_MAP" "$@"
        "$PYTHON" "$REFINE_SPELLING_MAP" "$@"
        "$PYTHON" "$MAKE_FASTTEXT_SLICES" "$@"
        "$PYTHON" "$EXTRACT_NEIGHBOURHOODS" "$@"
        "$PYTHON" "$ANNOTATE_NEIGHBOR_ROLES" "$@"
        "$PYTHON" "$BUILD_ROLE_PROFILES" "$@"
        "$PYTHON" "$VISUALISE" "$@"
        ;;
esac

cd "$OUR_OLDPWD"
