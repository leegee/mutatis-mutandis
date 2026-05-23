import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

from lib.eebo_config import MACBERTH_MODEL_PATH, CONCEPT_SETS


# -----------------------------
# CONFIG (LIVE MODE)
# -----------------------------

ACTIVE_CONCEPT = "LIBERTY"
N_PER_FORM = 5


# -----------------------------
# CONTEXT TEMPLATES
# (still intentionally uniform for now)
# -----------------------------

TEMPLATES = [
    "in this sentence the {word} is declared by authority",
    "the use of {word} appears in parliamentary discourse",
    "it is argued that {word} must be understood rightly",
    "according to the text the {word} is affirmed",
]


# -----------------------------
# OCR SIMULATION
# -----------------------------

def corrupt(word: str) -> str:
    rules = [
        lambda w: w.replace("i", "l"),
        lambda w: w.replace("e", "ee"),
        lambda w: w.replace("v", "u"),
        lambda w: w[:-1] if len(w) > 4 and random.random() < 0.2 else w,
        lambda w: "l" + w if random.random() < 0.3 else w,
    ]
    return random.choice(rules)(word)


# -----------------------------
# DATA BUILDER (SINGLE CONCEPT)
# -----------------------------

def build_dataset(concept_spec):
    dataset = defaultdict(list)

    forms = list(concept_spec["forms"])
    false_forms = list(concept_spec["false_positives"])

    for f in forms:
        for _ in range(N_PER_FORM):
            dataset["clean"].append(random.choice(TEMPLATES).format(word=f))
            dataset["ocr"].append(random.choice(TEMPLATES).format(word=corrupt(f)))

    for f in false_forms:
        for _ in range(max(1, N_PER_FORM // 2)):
            dataset["false"].append(random.choice(TEMPLATES).format(word=f))

    return dataset


# -----------------------------
# MODEL
# -----------------------------

def load_model():
    tok = AutoTokenizer.from_pretrained(MACBERTH_MODEL_PATH, local_files_only=True)
    model = AutoModel.from_pretrained(MACBERTH_MODEL_PATH, local_files_only=True)
    model.eval()
    return tok, model


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1)


def embed(texts, tok, model, device):
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, return_dict=True)

    return mean_pool(out.last_hidden_state, enc["attention_mask"]).cpu().numpy()


def centroid(v):
    v = np.mean(v, axis=0)
    return v / (np.linalg.norm(v) + 1e-12)


# -----------------------------
# LIVE ANALYSIS
# -----------------------------

def run():
    tok, model = load_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    spec = CONCEPT_SETS[ACTIVE_CONCEPT]
    data = build_dataset(spec)

    print("\n==============================")
    print("ACTIVE CONCEPT:", ACTIVE_CONCEPT)
    print("==============================\n")

    # embed each group
    grouped_vecs = {}

    for label, texts in data.items():
        vecs = embed(texts, tok, model, device)
        grouped_vecs[label] = vecs

        print(f"{label:6} n={len(texts)}")

    print("\n--- CENTROID SIMILARITIES ---\n")

    keys = list(grouped_vecs.keys())
    centroids = {k: centroid(v) for k, v in grouped_vecs.items()}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]

            sim = cosine_similarity(
                centroids[a].reshape(1, -1),
                centroids[b].reshape(1, -1)
            )[0, 0]

            print(f"{a:6} ↔ {b:6}: {sim:.4f}")

    print("\n--- INTERPRETATION ---")
    print("clean ↔ ocr   = OCR robustness")
    print("clean ↔ false = semantic boundary control")
    print("ocr ↔ false   = leakage / collapse signal")


if __name__ == "__main__":
    run()

