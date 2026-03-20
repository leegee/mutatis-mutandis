# import json
# import os
# from pathlib import Path
# from google.colab import drive

# # Mount Google Drive
# drive.mount('/content/drive', force_remount=True)

# # Define paths
# repo_base_path = Path("/content/mutatis-mutandis")
# python_dir = repo_base_path / "python"
# src_dir = python_dir / "src"

# # Change working directory to python
# os.chdir(python_dir)

# # Add python dir to PYTHONPATH so 'lib' can be imported
# os.environ["PYTHONPATH"] = str(python_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

# # Load Postgres credentials from a private JSON file on Drive
# creds_path = Path("/content/drive/MyDrive/macberth_pg_secrets.json")
# with open(creds_path) as f:
#     creds = json.load(f)

# os.environ["PGHOST"] = creds["host"]
# os.environ["PGPORT"] = creds.get("port", "5432")
# os.environ["PGDATABASE"] = creds["database"]
# os.environ["PGUSER"] = creds["user"]
# os.environ["PGPASSWORD"] = creds["password"]

# !echo $PGHOST:$PGPORT

# # Run the pipeline as a subprocess
# !python {src_dir / 'slice_embedding_pipeline.py'} --force



#
# Colab notebook to act as ./pipline.sh -p train
# and generate from MacBERTh embeddings on Google Drive
#

import subprocess
import sys
import os
import json
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


# Add to path
sys.path.append(str(python_dir / "src"))
from slice_embedding_pipeline import main

os.environ["EEBO_FORCE_OVERWRITE"] = "1"  # or "true"


# Call the pipeline directly
main()


# # Run the generation pipeline
# pipeline_script = src_dir / "slice_embedding_pipeline.py"
# try:
#     # Run with -u for unbuffered output to see real-time logs
#     result = subprocess.run(
#         ["python", "-u", str(pipeline_script), "--force"],
#         check=True,
#         stdout=sys.stdout,
#         stderr=sys.stderr
#     )
#     # If capture_output is False (default), result.stdout/stderr will be None
#     # The output would have streamed directly.
#     # These print statements are only relevant if capture_output=True was used.
#     print(result.stdout)
#     print(result.stderr)
# except subprocess.CalledProcessError as e:
#     print(f"Pipeline script failed with exit code {e.returncode}")
#     print("--- Standard Output ---")
#     print(e.stdout)
#     print("--- Standard Error ---")
#     print(e.stderr)
#     raise # Re-raise the exception to mark the cell as failed
