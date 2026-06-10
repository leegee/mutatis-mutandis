#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON_DIR="$SCRIPT_DIR/python"
SRC="$PYTHON_DIR/src"

# PYTHON="${PYTHON:-python}"
PYTHON="python"

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
    0|in|ingest)
        RUN_SCRIPTS+=("$SRC/tier0_0_eebo_parse_tei.py")
        RUN_ENVS+=("")
        ;;
    1|embed)
        RUN_SCRIPTS+=("$SRC/tier1_0_corpus2zarr.py")
        RUN_ENVS+=("")
        ;;
    2|ann)
        RUN_SCRIPTS+=("$SRC/tier1_5_build_faiss_index.py")
        RUN_ENVS+=("")
        ;;
    3|graph)
        RUN_SCRIPTS+=("$SRC/tier2_0_concept_events.py")
        RUN_ENVS+=("")
        ;;
    4|plot)
        RUN_SCRIPTS+=("$SRC/tier3_0_plots.py")
        RUN_ENVS+=("")
        ;;
    all)
        RUN_SCRIPTS+=(
            "$SRC/tier1_0_corpus2zarr.py --clear"
            "$SRC/tier1_5_build_faiss_index.py  --clear"
            "$SRC/tier2_0_concept_events.py  --clear"
            "$SRC/tier3_0_plots.py  --clear"
        )
        RUN_ENVS+=("" "" "")
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
# command -v ruff >/dev/null || { echo "ruff not installed"; popd >/dev/null; exit 1; }
# command -v mypy >/dev/null || { echo "mypy not installed"; popd >/dev/null; exit 1; }
# command -v pyright >/dev/null || { echo "pyright not installed"; popd >/dev/null; exit 1; }
#
# echo "# Running Ruff"
# ruff check "$SRC"
#
# echo "# Running Mypy"
# mypy "$SRC"
#
# echo "# Running Pyright"
# pyright "$SRC"
#
# echo "# All checks passed"

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
