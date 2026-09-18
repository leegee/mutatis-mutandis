# ============================================================
# Colab notebook – MacBERTh medium-window embedder
# Writes Parquet files to Google Drive (later ingested into Lance)
#
# Update `./macberth_pg_secrets.json` on Google Drive's root dir
# with the host/port output from `ngrok tcp 5432`.
#
# Do not forget to restart the Colab session when the IP changes.
#
#
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

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "uv",
    ],
    check=True,
)

print("Installing project dependencies from pyproject.toml / uv.lock...")

subprocess.run(
    [
        "uv",
        "sync",
        "--directory",
        str(python_dir),
    ],
    check=True,
)

print("Dependencies installed.")

# ------------------------------------------------------------
# 4. Postgres credentials (from Drive)
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
# 5. Make sure our package is importable
# ------------------------------------------------------------
sys.path.insert(0, str(src_dir))

# ------------------------------------------------------------
# 6. Run the window embedder
# ------------------------------------------------------------
# Optional: populate the job queue first (only needed once)
# Uncomment the next two lines if the jobs table is empty.
#
# print("Populating embedding_jobs ...")
# subprocess.run([
#     sys.executable, "-m", "tier1.tier1_corpus2events",
#     "--populate",
#     # "--corpus", "eebo",
#     # "--min-year", "1600",
#     # "--max-year", "1700",
# ], check=True)

print("Starting window embedder worker (Colab → Parquet on Drive) ...")

subprocess.run([
    "uv", "run",
    "--directory", str(python_dir),
    "-m", "tier1.tier1_corpus2events",
    "--backend", "onnx",
    # "--max-docs", "5",          # useful for a quick test
    # "--dry-run",                # embed only, write nothing
    # "--worker-id", "colab-1",   # optional explicit id
], check=True)

print("Done.")
