#!/usr/bin/env bash
# Minimal Git Bash activation shim

VENV_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"

export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/Scripts:$PATH"

echo "Activated venv at: $VIRTUAL_ENV"
