#
# Colab notebook to act as ./pipline.sh -p train
# and generate from MacBERTh embeddings on Google Drive
#
import subprocess
import json
import os
from pathlib import Path
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# Set repo paths
repo_base_path = Path("/content/mutatis-mutandis")
python_dir = repo_base_path / "python"
src_dir = python_dir / "src"

# Clone or update repo
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

# Change working directory and PYTHONPATH
os.chdir(python_dir)
os.environ["PYTHONPATH"] = str(python_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

# Install Python dependencies from requirements.txt
print("Installing Python dependencies from requirements.txt...")
subprocess.run(["pip", "install", "-r", str(python_dir / "requirements.txt")], check=True)
print("Dependencies installed.")

# Load Postgres credentials from Drive
creds_path = Path("/content/drive/MyDrive/macberth_pg_secrets.json")
with open(creds_path) as f:
    creds = json.load(f)

os.environ["PGHOST"] = creds["host"]
os.environ["PGPORT"] = creds.get("port")
os.environ["PGDATABASE"] = creds["database"]
os.environ["PGUSER"] = creds["user"]
os.environ["PGPASSWORD"] = creds["password"]

print(f"Postgres target: {os.environ['PGHOST']}:{os.environ['PGPORT']}")

# Define persistent Drive output folder
MACBERTH_OUTPUT_DIR = Path("/content/drive/MyDrive/macberth_output")

# Update paths for Colab to persist output in Drive
import lib.eebo_config as cfg

cfg.MACBERTH_ALIGNED_VECTORS_DIR = MACBERTH_OUTPUT_DIR / "aligned_vectors"
cfg.MACBERTH_SLICE_MODEL_DIR = MACBERTH_OUTPUT_DIR / "slices"
cfg.MACBERTH_FINE_TUNED_DIR = MACBERTH_OUTPUT_DIR / "finetuned"

# Make sure directories exist
for p in [cfg.MACBERTH_ALIGNED_VECTORS_DIR, cfg.MACBERTH_SLICE_MODEL_DIR, cfg.MACBERTH_FINE_TUNED_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Run the generation pipeline
pipeline_script = src_dir / "slice_embedding_pipeline.py"
try:
    # Capture stdout and stderr to get more detailed error messages
    result = subprocess.run(
        ["python", str(pipeline_script), "--force"],
        check=True,
        # capture_output=True,
        text=True # Decode stdout/stderr as text
    )
    print(result.stdout)
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Pipeline script failed with exit code {e.returncode}")
    print("--- Standard Output ---")
    print(e.stdout)
    print("--- Standard Error ---")
    print(e.stderr)
    raise # Re-raise the exception to mark the cell as failed
