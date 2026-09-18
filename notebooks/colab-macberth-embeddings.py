# ============================================================
# Colab notebook – MacBERTh medium-window embedder
# ============================================================

import subprocess
import sys
import os
import json
from pathlib import Path
from google.colab import drive

# ------------------------------------------------------------
# 1. Mount Drive
# ------------------------------------------------------------
drive.mount('/content/drive', force_remount=True)

# ------------------------------------------------------------
# 2. Repo setup
# ------------------------------------------------------------
repo_base_path = Path("/content/mutatis-mutandis")
python_dir = repo_base_path / "python"
src_dir = python_dir / "src"

if not repo_base_path.exists():
    print(f"Cloning repository into {repo_base_path} ...")
    subprocess.run([
        "git", "clone",
        "https://github.com/leegee/mutatis-mutandis.git",
        str(repo_base_path)
    ], check=True)
else:
    print(f"Pulling latest changes in {repo_base_path} ...")
    subprocess.run(["git", "-C", str(repo_base_path), "pull"], check=True)

os.chdir(python_dir)
os.environ["PYTHONPATH"] = str(src_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

# ------------------------------------------------------------
# 3. Install dependencies with uv
# ------------------------------------------------------------
print("Installing uv...")
subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)

print("Installing project dependencies...")
subprocess.run(["uv", "sync", "--directory", str(python_dir)], check=True)
print("Dependencies installed.")

# ------------------------------------------------------------
# 3b. Ensure MacBERTh model weights are present
# ------------------------------------------------------------
MODEL_DIR = src_dir / "lib" / "macberth-huggingface"
DRIVE_TGZ = Path("/content/drive/MyDrive/macberth_models/macberth-huggingface.tar.gz")

print("Looking for model at:", MODEL_DIR)
print("Drive archive at:", DRIVE_TGZ, "exists =", DRIVE_TGZ.exists())

if not (MODEL_DIR / "config.json").exists():
    if not DRIVE_TGZ.exists():
        raise FileNotFoundError(
            f"Model archive not found on Drive: {DRIVE_TGZ}\n"
            "Upload macberth-huggingface.tar.gz to that location first."
        )
    print("Extracting MacBERTh model from Drive ...")
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", str(DRIVE_TGZ), "-C", str(MODEL_DIR.parent)],
        check=True,
    )
    print("Extraction finished.")
else:
    print("Model already present.")

# Sanity check
required = ["config.json"]
# accept either weight format
has_weights = (
    (MODEL_DIR / "pytorch_model.bin").exists()
    or (MODEL_DIR / "model.safetensors").exists()
)
if not has_weights or any(not (MODEL_DIR / f).exists() for f in required):
    print("Contents of model dir:")
    for p in sorted(MODEL_DIR.iterdir()):
        print(" ", p.name)
    raise FileNotFoundError("Model files incomplete after extraction")


# ------------------------------------------------------------
# 3c. Prefer GPU ONNX Runtime when a GPU is present
# ------------------------------------------------------------
print("Checking for GPU ...")
gpu_available = False
try:
    import torch
    gpu_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available() = {gpu_available}")
    if gpu_available:
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Could not query torch CUDA:", e)

if gpu_available:
    print("Installing onnxruntime-gpu ...")
    # Remove the CPU-only package first if it was pulled in by uv
    subprocess.run(
        ["uv", "pip", "uninstall", "onnxruntime", "-y"],
        check=False,          # ignore if it wasn't installed
    )
    subprocess.run(
        [
            "uv", "pip", "install",
            "onnxruntime-gpu",
            "--directory", str(python_dir),
        ],
        check=True,
    )
    print("onnxruntime-gpu installed.")
else:
    print("No GPU detected – staying with CPU onnxruntime.")


import onnxruntime as ort
print("ORT available providers:", ort.get_available_providers())

# ------------------------------------------------------------
# 4. Postgres credentials
# ------------------------------------------------------------
creds_path = Path("/content/drive/MyDrive/macberth_pg_secrets.json")
with open(creds_path) as f:
    creds = json.load(f)

os.environ["PGHOST"]     = creds["host"]
os.environ["PGPORT"]     = str(creds.get("port", 5432))
os.environ["PGDATABASE"] = creds["database"]
os.environ["PGUSER"]     = creds["user"]
os.environ["PGPASSWORD"] = creds["password"]

print(f"Postgres target: {os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}")

# ------------------------------------------------------------
# 5. Force Colab mode for the child process
# ------------------------------------------------------------
# This environment variable is the reliable way to tell the
# subprocess that it is running on Colab.
os.environ["COLAB_MODE"] = "1"

# ------------------------------------------------------------
# 6. Run the window embedder
# ------------------------------------------------------------
print("Starting window embedder worker (Colab → Parquet on Drive) ...")

try:
    result = subprocess.run(
        [
            "uv", "run",
            "--directory", str(python_dir),
            "-m", "tier1.tier1_corpus2events",
            "--backend", "auto",
            # "--max-docs", "5",
            # "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,          # pass the COLAB_MODE=1 we just set
    )
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
    print(f"Stdout:\n{e.stdout}")
    print(f"Stderr:\n{e.stderr}")
    raise

print("Done.")
