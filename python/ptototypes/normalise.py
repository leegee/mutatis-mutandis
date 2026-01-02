import subprocess
from pprint import pprint
import os
import torch

# Local MacBERTh
# Replace this with your actual tagging head when available.
from macberth_pipe.tokenizer import MacBERThTokenizer
from macberth_pipe.network import MacBERThModel


# VARD2 CONFIG
VARD_DIR = "../lib/VARD2.5.4"
JARS = [
    f"{VARD_DIR}/vardstdin.jar",
    f"{VARD_DIR}/model.jar",
    f"{VARD_DIR}/clui.jar"
]

CP_SEP = ";" if os.name == "nt" else ":"
CLASSPATH = CP_SEP.join(JARS)

VARD_MAIN_CLASS = "vardstdin.VARDMain"
SETUP_FOLDER = f"{VARD_DIR}/setup"
THRESHOLD = "80"
FSCORE = "1.0"
USE_CACHE = "true"


# LOCAL MacBERTh TOKEN-CLASSIFIER
def load_local_tagger(
    vocab="./macberth_local/vocab.json",
    model_path="./macberth_local/macberth-small.pt",
    device="cpu"
):
    """
    Local MacBERTh tagger: no HuggingFace, no downloads.

    For now, the network lacks a tagging head, so we return
    per-token embeddings and wrap them in a simple format.
    """

    tokenizer = MacBERThTokenizer(vocab)
    model = MacBERThModel.load(model_path, map_location=device)
    model.eval()

    return tokenizer, model


tokenizer, tagger_model = load_local_tagger()


def local_tag(text: str):
    """
    Local replacement for the HuggingFace pipeline("token-classification").

    Until you define a real tagger head, this returns token embeddings
    and dummy labels like: {'token': tok, 'label': 'EMB', 'vector': embedding}

    You can later replace this with a CRF, MLP or classification head.
    """
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text)
    t = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        out = tagger_model(t)
        emb = out["emb"].squeeze(0)   # (T, D)

    results = []
    for i, tok in enumerate(tokens):
        results.append({
            "token": tok,
            "label": "EMB",  # placeholder
            "vector": emb[i].cpu().numpy().tolist()
        })

    return results


# VARD NORMALISATION
def normalize_with_vard(text: str) -> str:
    """Send text to VARD2 over STDIN and capture normalized output."""
    try:
        args = [
            "java",
            "-cp", CLASSPATH,
            VARD_MAIN_CLASS,
            SETUP_FOLDER,
            THRESHOLD,
            FSCORE,
            "-",          # read stdin
            "false",      # no folder recursion
            "-",          # write stdout
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
            return text
        return stdout.strip()
    except Exception as e:
        print("Exception while running VARD2:", e)
        return text


# PIPELINE

def normalize_and_classify(text: str):
    normalized_text = normalize_with_vard(text)
    classified = local_tag(normalized_text)
    return {
        "original": text,
        "normalized": normalized_text,
        "classified": classified
    }


# EG

if __name__ == "__main__":
    sample_text = "Thys is a pamphlet sentence."
    result = normalize_and_classify(sample_text)
    pprint(result)
