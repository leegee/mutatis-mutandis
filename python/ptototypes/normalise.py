import subprocess
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os
import sys

# --- Paths & model ---
VARD_DIR = "./VARD2.5.4"
JARS = [
    f"{VARD_DIR}/vardstdin.jar",
    f"{VARD_DIR}/model.jar",
    f"{VARD_DIR}/clui.jar"
]

# Platform-specific classpath separator
CP_SEP = ";" if os.name == "nt" else ":"
CLASSPATH = CP_SEP.join(JARS)

VARD_MAIN_CLASS = "vardstdin.VARDMain"  # main class for stdin mode
SETUP_FOLDER = f"{VARD_DIR}/setup"      # VARD2 setup folder
THRESHOLD = "80"                        # threshold (0-100)
FSCORE = "1.0"                          # f-score weight
USE_CACHE = "true"                       # normalization cache

MODEL_NAME = "emanjavacas/MacBERTh"

# Initialize MacBERTh
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
nlp = pipeline("token-classification", model=model, tokenizer=tokenizer, device=-1)

# --- Function to normalize text using VARD2 via STDIN/STDOUT ---
def normalize_with_vard(text: str) -> str:
    """Send text to VARD2 over STDIN and capture normalized output."""
    try:
        # VARD2 arguments: setup_folder threshold fscore_weight input read_subfolders output use_cache
        args = [
            "java",
            "-cp", CLASSPATH,
            VARD_MAIN_CLASS,
            SETUP_FOLDER,
            THRESHOLD,
            FSCORE,
            "-",          # "-" means read from stdin
            "false",      # don't search subfolders
            "-",          # "-" means write to stdout
            USE_CACHE
        ]
        proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = proc.communicate(input=text)
        if proc.returncode != 0:
            print("VARD2 failed!")
            print("stderr:", stderr)
            return text  # fallback
        return stdout.strip()
    except Exception as e:
        print("Exception while running VARD2:", e)
        return text  # fallback

# --- Combined pipeline ---
def normalize_and_classify(text: str):
    normalized_text = normalize_with_vard(text)
    classified = nlp(normalized_text)
    return {"original": text, "normalized": normalized_text, "classified": classified}

# --- Example usage ---
if __name__ == "__main__":
    sample_text = "Thys is a pamphlet sentence."
    result = normalize_and_classify(sample_text)
    pprint(result)
