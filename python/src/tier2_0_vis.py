"""
tier2_0_vis.py - Tier2 Concept Visualisation Toolkit
----------------------------------------------------

This script visualises output from tier2_concept_neighbours.json
produced by tier2_0_concept_events.py.

It provides multiple complementary views:

1. Token frequency distribution (aggregate neighbours)
2. Document distribution
3. Window distribution heatmap-like aggregation
4. Neighbour score distribution
5. Concept co-occurrence graph (token graph)

Assumes JSON structure:
{
  concept_name: {
    concept,
    n_events,
    aggregate: {top_tokens, top_docs, top_windows},
    events: [
      {
        event_id,
        token,
        neighbours: [
          {token, doc_id, score, ...}
        ]
      }
    ]
  }
}
"""

import json
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from tier2_0_concept_events import OUTPUT_PATH as INPUT_PATH


def load_data(path=INPUT_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# 1. Token frequency
# ------------------------------------------------------------

def plot_top_tokens(concept_data, top_k=20):
    tokens = Counter()

    for event in concept_data["events"]:
        for n in event["neighbours"]:
            tokens[n["token"]] += 1

    most = tokens.most_common(top_k)
    labels, values = zip(*most) if most else ([], [])

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top neighbour tokens")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 2. Document distribution
# ------------------------------------------------------------

def plot_doc_distribution(concept_data, top_k=20):
    docs = Counter()

    for event in concept_data["events"]:
        for n in event["neighbours"]:
            docs[n["doc_id"]] += 1

    most = docs.most_common(top_k)
    labels, values = zip(*most) if most else ([], [])

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top neighbour documents")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 3. Window distribution (doc, window)
# ------------------------------------------------------------

def plot_window_distribution(concept_data, top_k=20):
    windows = Counter()

    for event in concept_data["events"]:
        for n in event["neighbours"]:
            key = f"{n['doc_id']}::{n['window_id']}"
            windows[key] += 1

    most = windows.most_common(top_k)
    labels, values = zip(*most) if most else ([], [])

    plt.figure()
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top (doc, window) neighbourhood density")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. Score distribution
# ------------------------------------------------------------

def plot_score_distribution(concept_data):
    scores = []

    for event in concept_data["events"]:
        for n in event["neighbours"]:
            scores.append(n["score"])

    plt.figure()
    plt.hist(scores, bins=50)
    plt.title("Neighbour similarity score distribution")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5. Token graph (co-occurrence in neighbourhoods)
# ------------------------------------------------------------

def plot_token_graph(concept_data, min_edge_weight=3, max_nodes=50):
    G = nx.Graph()

    edge_weights = Counter()

    for event in concept_data["events"]:
        tokens = [n["token"] for n in event["neighbours"]]

        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                a, b = sorted((tokens[i], tokens[j]))
                edge_weights[(a, b)] += 1

    for (a, b), w in edge_weights.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    if len(G.nodes) > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n, _ in top_nodes}
        G = G.subgraph(keep)

    plt.figure(figsize=(10, 8), facecolor="black")
    ax = plt.gca()
    ax.set_facecolor("black")

    pos = nx.spring_layout(G, seed=42)

    widths = [G[u][v]["weight"] for u, v in G.edges]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=120,
        node_color="white",
        alpha=0.9
    )

    nx.draw_networkx_edges(
        G,
        pos,
        width=widths,
        edge_color="gray",
        alpha=0.4
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=7,
        font_color="white"
    )

    plt.title("Neighbour token co-occurrence graph", color="white")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------

def run_all(concept_name):
    data = load_data()

    if concept_name not in data:
        raise ValueError(f"Concept {concept_name} not found")

    concept_data = data[concept_name]

    print(f"Concept: {concept_name}")
    print(f"Events: {concept_data.get('n_events')}")

    plot_top_tokens(concept_data)
    plot_doc_distribution(concept_data)
    plot_window_distribution(concept_data)
    plot_score_distribution(concept_data)
    plot_token_graph(concept_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--concept", type=str, required=True)
    args = parser.parse_args()

    run_all(args.concept.upper())
