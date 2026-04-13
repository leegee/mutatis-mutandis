#!/usr/bin/env python
from mb_test import OUT_PATH

import json
import math
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

from lib.eebo_logging import logger


HOVER_FONT_SIZE = 48
CENTER_FONT_SIZE = 32
PLOT_FONT_SIZE = 18

HIGH_CONTRAST_COLORS = [
    "#FF3333", "#33FF33", "#3333FF",
    "#FFFF33", "#FF33FF",
    "#33FFFF", "#FFA533", "#800080", "#338000", "#3333AA",
    "#FFC0CB", "#808000"
]


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


data = load_data(OUT_PATH)
tokens = list(data.keys())


def slice_keys(token_data):
    # invariant: slices are time-ordered after pipeline sort
    return [s["slice_start"] for s in token_data.get("slices", [])]


def get_slice(token_data, slice_start):
    # failure mode: missing slice → fallback to last known state
    slices = token_data.get("slices", [])
    return next((s for s in slices if s["slice_start"] == slice_start), slices[-1] if slices else None)


def normalize_series(series):
    lo, hi = min(series), max(series)
    if hi <= lo:
        return [0.0] * len(series)
    return [(x - lo) / (hi - lo) for x in series]


def create_main_dashboard_figure(data, normalize=False, highlight_slice=None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=["Structural Drift", "Distributional Drift"]
    )

    for idx, (token, token_data) in enumerate(data.items()):
        color = HIGH_CONTRAST_COLORS[idx % len(HIGH_CONTRAST_COLORS)]
        slices = token_data.get("slices", [])
        if not slices:
            continue

        xs = [s["slice_start"] for s in slices]
        drift = [s.get("drift", 0) for s in slices]
        jsd = [s.get("js_divergence", 0) for s in slices]

        if normalize:
            drift = normalize_series(drift)
            jsd = normalize_series(jsd)

        fig.add_trace(go.Scatter(
            x=xs, y=drift,
            mode="lines+markers",
            name=token,
            legendgroup=token,
            marker=dict(color=color),
            customdata=[{"token": token, "color": color}] * len(xs)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=xs, y=jsd,
            mode="lines+markers",
            name=token,
            legendgroup=token,
            marker=dict(color=color),
            line=dict(dash="dot"),
            showlegend=False
        ), row=2, col=1)

        if highlight_slice in xs:
            i = xs.index(highlight_slice)
            fig.add_trace(go.Scatter(
                x=[xs[i]], y=[drift[i]],
                mode="markers",
                marker=dict(size=14, color="yellow"),
                showlegend=False
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=[xs[i]], y=[jsd[i]],
                mode="markers",
                marker=dict(size=14, color="yellow"),
                showlegend=False
            ), row=2, col=1)

    fig.update_layout(
        title="Semantic Drift Decomposition",
        template="plotly_dark",
        height=900,
        hoverlabel=dict(font=dict(size=HOVER_FONT_SIZE)),
        uirevision="constant"
    )
    return fig


def create_neighbor_figure(token, neighbors, slice_start, base_color):
    if not neighbors:
        return go.Figure()

    slices = data[token].get("slices", [])
    slice_data = get_slice(data[token], slice_start)
    drift_mag = slice_data.get("drift", 0.0) if slice_data else 0.0

    drift_scale = 0.5 + 2.0 * drift_mag

    G = nx.Graph()
    G.add_node(token, sim=1.0)

    for n in neighbors:
        t = n.get("token")
        sim = n.get("similarity", 0)
        if t:
            G.add_node(t, sim=sim)
            G.add_edge(token, t, weight=1 - sim)

    pos = nx.spring_layout(G, weight="weight", seed=42)

    cx, cy = pos[token]
    for n in pos:
        if n != token:
            dx, dy = pos[n][0] - cx, pos[n][1] - cy
            pos[n] = (cx + dx * drift_scale, cy + dy * drift_scale)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, sizes, labels, tips = [], [], [], [], []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        sim = G.nodes[n].get("sim", 0)

        if n == token:
            sizes.append(CENTER_FONT_SIZE)
            labels.append(n)
            tips.append(f"{n} (drift={drift_mag:.2f})")
        else:
            size = 12 + 18 * (1 - sim)
            sizes.append(size)
            labels.append(n)
            tips.append(f"{n} (sim={sim:.2f})")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(color="gray", width=1),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        marker=dict(size=sizes, color=base_color),
        text=tips,
        hoverinfo="text",
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="text",
        text=labels,
        textposition="middle center",
        textfont=dict(size=PLOT_FONT_SIZE),
        hoverinfo="none",
        showlegend=False
    ))

    fig.update_layout(
        title=f"Neighbors of '{token}' ({slice_start})",
        template="plotly_dark",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=800,
        width=1200,
        hoverlabel=dict(font=dict(size=HOVER_FONT_SIZE)),
        uirevision="constant"
    )

    return fig


# ---------------- Dash ----------------

app = dash.Dash(__name__)

all_slices = sorted({
    s["slice_start"]
    for t in data.values()
    for s in t.get("slices", [])
})


app.layout = html.Div([
    html.H1("EEBO Semantic Drift Dashboard"),

    dcc.Graph(id="main-dashboard", figure=create_main_dashboard_figure(data)),

    dcc.Dropdown(
        id="token-dropdown",
        options=[{"label": t, "value": t} for t in sorted(tokens)],
        value=tokens[0] if tokens else None
    ),

    dcc.Slider(
        id="slice-slider",
        min=min(all_slices) if all_slices else 0,
        max=max(all_slices) if all_slices else 0,
        step=None,
        marks={s: str(s) for s in all_slices},
        value=all_slices[0] if all_slices else None
    ),

    dcc.Graph(id="neighbor-graph"),

    dcc.Store(id="selected", data={"token": None, "slice": None})
])


@app.callback(
    Output("selected", "data"),
    Input("main-dashboard", "clickData"),
    Input("token-dropdown", "value"),
    Input("slice-slider", "value"),
    State("selected", "data")
)
def update_selected(clickData, token, slice_start, current):
    if not current:
        current = {}

    if token:
        current["token"] = token
    if slice_start:
        current["slice"] = slice_start

    if clickData:
        pt = clickData["points"][0]
        cd = pt.get("customdata") or {}
        if cd.get("token"):
            current["token"] = cd["token"]
        if pt.get("x"):
            current["slice"] = pt["x"]

    return current


@app.callback(
    Output("neighbor-graph", "figure"),
    Input("selected", "data")
)
def update_neighbors(selected):
    if not selected or not selected.get("token"):
        return go.Figure()

    token = selected["token"]
    slice_start = selected.get("slice")

    token_data = data.get(token, {})
    slice_data = get_slice(token_data, slice_start)

    if not slice_data:
        return go.Figure()

    neighbors = slice_data.get("top_neighbors", [])

    return create_neighbor_figure(
        token,
        neighbors,
        slice_start,
        base_color="#225"
    )


if __name__ == "__main__":
    app.run(debug=True)
