#!/usr/bin/env python
"""
Tier2 Graph Explorer (Dash Prototype) - Multi-Concept Edition
------------------------------------------------------------

Upgrades:
- Concept selector (multi-graph support)
- Per-concept stateful node removal
- Stable layout per concept
- Toggle node removal (click again restores)
- Min edge weight + max nodes filters
- Clean separation of graph state per concept

This remains a prototype intended for SolidJS migration later.
"""

import json
from collections import Counter

import networkx as nx

from tier2_0_concept_events import OUTPUT_PATH as INPUT_PATH

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

def load_data():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Graph building
# ------------------------------------------------------------

def build_graph(concept_data, min_edge_weight=3):
    edge_weights = Counter()

    for event in concept_data["events"]:
        tokens = [n["token"] for n in event["neighbours"]]

        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                a, b = sorted((tokens[i], tokens[j]))
                edge_weights[(a, b)] += 1

    G = nx.Graph()

    for (a, b), w in edge_weights.items():
        if w >= min_edge_weight:
            G.add_edge(a, b, weight=w)

    return G


def subset_graph(G, max_nodes, removed):
    if removed:
        G = G.subgraph([n for n in G.nodes if n not in removed]).copy()

    if len(G.nodes) > max_nodes:
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        keep = {n for n, _ in top_nodes}
        G = G.subgraph(keep).copy()

    return G


# ------------------------------------------------------------
# Layout (stable per concept graph instance)
# ------------------------------------------------------------

def compute_layout(G):
    return nx.spring_layout(G, seed=42)


# ------------------------------------------------------------
# Plot builder
# ------------------------------------------------------------

def make_figure(G, pos):
    edge_x, edge_y = [], []

    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(220,220,220,0.55)"),
        hoverinfo="none"
    )

    node_x, node_y, labels = [], [], []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        labels.append(n)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=10, color="white"),
        textfont=dict(color="white"),
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    return fig


# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------

data = load_data()
concept_keys = list(data.keys())

app = Dash(__name__)


app.layout = html.Div(
    style={"backgroundColor": "black", "height": "100vh"},
    children=[

        html.H3("Tier2 Multi-Concept Graph Explorer", style={"color": "white"}),

        # ---------------- concept selector ----------------
        html.Div([
            html.Label("Concept", style={"color": "white"}),
            dcc.Dropdown(
                id="concept",
                options=[{"label": k, "value": k} for k in concept_keys],
                value=concept_keys[0]
            )
        ]),

        # ---------------- controls ----------------
        html.Div([
            html.Label("Max nodes", style={"color": "white"}),
            dcc.Dropdown(
                id="max-nodes",
                options=[
                    {"label": "20", "value": 20},
                    {"label": "50", "value": 50},
                    {"label": "100", "value": 100},
                ],
                value=50
            ),

            html.Label("Min edge weight", style={"color": "white"}),
            dcc.Slider(
                id="min-edge",
                min=1,
                max=10,
                step=1,
                value=3
            ),
        ]),

        dcc.Store(id="removed-nodes", data={}),  # per-concept state

        dcc.Graph(
            id="graph",
            style={"height": "88vh"}
        ),

        html.Div("Click node to toggle removal (per concept)", style={"color": "gray"})
    ]
)


# ------------------------------------------------------------
# CALLBACK
# ------------------------------------------------------------

@app.callback(
    Output("graph", "figure"),
    Output("removed-nodes", "data"),
    Input("concept", "value"),
    Input("max-nodes", "value"),
    Input("min-edge", "value"),
    Input("graph", "clickData"),
    State("removed-nodes", "data"),
)
def update_graph(concept, max_nodes, min_edge, clickData, removed_map):

    removed_map = removed_map or {}
    removed = set(removed_map.get(concept, []))

    # toggle node
    if clickData and "points" in clickData:
        node = clickData["points"][0].get("text")
        if node:
            if node in removed:
                removed.remove(node)
            else:
                removed.add(node)

    removed_map[concept] = list(removed)

    concept_data = data[concept]

    G = build_graph(concept_data, min_edge_weight=min_edge)
    G = subset_graph(G, max_nodes, removed)

    pos = compute_layout(G)
    fig = make_figure(G, pos)

    return fig, removed_map


# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
