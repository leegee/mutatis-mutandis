import json
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from tier2_0_concept_events import OUTPUT_PATH as INPUT_PATH

path = Path(INPUT_PATH)

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

def get_concept(name):
    return data[name]

def plot_top_tokens(concept_name, n=20):
    concept = get_concept(concept_name)
    tokens = concept["aggregate"]["top_tokens"]

    labels = [t[0] for t in tokens[:n]]
    values = [t[1] for t in tokens[:n]]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top neighbour tokens: {concept_name}")
    plt.tight_layout()
    plt.show()


def plot_top_docs(concept_name, n=20):
    concept = get_concept(concept_name)
    docs = concept["aggregate"]["top_docs"]

    labels = [d[0] for d in docs[:n]]
    values = [d[1] for d in docs[:n]]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top documents: {concept_name}")
    plt.tight_layout()
    plt.show()

def plot_window_distribution(concept_name, n=20):
    concept = get_concept(concept_name)
    windows = concept["aggregate"]["top_windows"]

    labels = [str(w[0]) for w in windows[:n]]
    values = [w[1] for w in windows[:n]]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Window distribution: {concept_name}")
    plt.tight_layout()
    plt.show()

concept = "LIBERTY"

plot_top_tokens(concept)
plot_top_docs(concept)
plot_window_distribution(concept)