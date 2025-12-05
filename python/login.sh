#!/usr/bin/env bash
set -e

VENV_DIR="./macberth_env"
STAMP="$VENV_DIR/.installed"

# --- 1. Create venv if missing ---
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "✔ Virtual environment already exists."
fi

# --- 2. Detect platform + activate ---
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Windows (Git Bash, Cygwin, MSYS)
    echo "🔌 Activating Windows venv..."
    source "$VENV_DIR/Scripts/activate"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # Linux / macOS
    echo "🔌 Activating UNIX venv..."
    source "$VENV_DIR/bin/activate"
else
    echo "❌ ERROR: Could not find activate script. The venv may be corrupted."
    exit 1
fi

# --- 3. Upgrade pip safely ---
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# --- 4. Install requirements only on first run ---
if [[ ! -f "$STAMP" ]]; then
    echo "📦 Installing dependencies from requirements.txt..."
    python -m pip install -r requirements.txt
    touch "$STAMP"
    echo "✔ Dependencies installed."
else
    echo "✔ Dependencies already installed. Skipping."
fi

echo "🎉 Environment ready!"
