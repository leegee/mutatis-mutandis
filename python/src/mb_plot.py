#!/usr/bin/env python
from mb_test import OUT_PATH

import json
import math
from pathlib import Path
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

def normalize_series(series):
    lo, hi = min(series), max(series)
    if hi <= lo:
        return [0.0]*len(series)
    return [(x - lo) / (hi - lo) for x in series]

data = load_data(OUT_PATH)
tokens = list(data.keys())
all_years = sorted({s["year"] for t in data.values() for s in t.get("slices", [])})

def create_main_dashboard_figure(data, normalize=False, highlight_year=None, base_color="#225"):
    title_suffix = " (Normalized)" if normalize else " (Raw)"
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[f"Structural Drift{title_suffix}", f"Distributional Drift{title_suffix}"]
    )

    for idx, (token, token_data) in enumerate(data.items()):
        color = HIGH_CONTRAST_COLORS[idx % len(HIGH_CONTRAST_COLORS)]
        slices = token_data.get("slices", [])
        if not slices:
            continue

        years = [s["year"] for s in slices]
        drift = [s.get("drift",0) for s in slices]
        jsd = [s.get("js_divergence",0) for s in slices]

        if normalize:
            drift = normalize_series(drift)
            jsd = normalize_series(jsd)

        fig.add_trace(go.Scatter(
            x=years, y=drift,
            mode='lines+markers',
            name=token,
            legendgroup=token,
            customdata=[{"token": token, "color": color}] * len(years),
            marker=dict(color=color)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=years, y=jsd,
            mode='lines+markers',
            name=token,
            legendgroup=token,
            customdata=[{"token": token, "color": color}] * len(years),
            line=dict(dash='dot'),
            marker=dict(color=color),
            showlegend=False
        ), row=2, col=1)

        if highlight_year in years:
            i = years.index(highlight_year)
            fig.add_trace(go.Scatter(
                x=[years[i]], y=[drift[i]],
                mode='markers',
                marker=dict(size=14, color='yellow'),
                showlegend=False
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[years[i]], y=[jsd[i]],
                mode='markers',
                marker=dict(size=14, color='yellow'),
                showlegend=False
            ), row=2, col=1)

        transitions = token_data.get("phase_transitions",{})
        for t in transitions.get("major",[]):
            for r in (1,2):
                fig.add_vline(x=t["year"], line_width=3, line_color="red", opacity=0.7, row=r, col=1)
        for t in transitions.get("minor",[]):
            for r in (1,2):
                fig.add_vline(x=t["year"], line_width=1, line_dash="dash", line_color="orange", opacity=0.6, row=r, col=1)

    fig.update_layout(
        title="Semantic Drift Decomposition",
        template="plotly_dark",
        height=900,
        hoverlabel=dict(font=dict(size=HOVER_FONT_SIZE)),
        uirevision="constant"
    )
    return fig


def create_neighbor_figure(token, neighbors, year, base_color):
    """
    Create a Plotly figure showing the neighbors of a given token.

    Args:
        token (str): The central token.
        neighbors (list of dict): Each dict has 'token', 'similarity', 'count'.
        year (int): Year of the slice.
        base_color (str): Base color to use for neighbor nodes.

    Returns:
        go.Figure: Plotly figure of the neighbor graph.
    """
    if not neighbors:
        return go.Figure()

    slices = data[token].get("slices", [])
    slice_for_year = next((s for s in slices if s["year"]==year), slices[-1])
    drift_mag = slice_for_year.get("drift", 0.0)

    drift_scale = 0.5 + 2.0 * drift_mag  # exaggerate neighbors if drift is large

    # Build graph
    G = nx.Graph()
    G.add_node(token, sim=1.0, count=1)  # central token
    for n in neighbors:
        t = n.get("token")
        sim = n.get("similarity", 0)
        count = n.get("count", 1)
        if t:
            G.add_node(t, sim=sim, count=count)
            G.add_edge(token, t, weight=1-sim)

    # Layout
    pos = nx.spring_layout(G, weight='weight', seed=42)

    # Scale neighbors by drift
    cx, cy = pos[token]
    for n in pos:
        if n != token:
            dx, dy = pos[n][0]-cx, pos[n][1]-cy
            pos[n] = (cx + dx*drift_scale, cy + dy*drift_scale)

    # Extract edges
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    # Extract nodes
    node_x, node_y, sizes, labels, tips = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        sim = G.nodes[n].get('sim', 0)
        count = G.nodes[n].get('count', 1)

        if n == token:
            sizes.append(CENTER_FONT_SIZE)  # central token big
            # labels.append(f"{n} (drift={drift_mag:.2f})")
            labels.append(f"{n}")
            tips.append(f"{n} (drift={drift_mag:.2f})")
        else:
            w = G[token][n]['weight']
            size = 12 + 18*(1-w) + math.log1p(count)*4  # similarity + count
            sizes.append(size)
            # labels.append(f"{n} (sim={sim:.2f}, count={count})")
            labels.append(f"{n}")
            tips.append(f"{n} (sim={sim:.2f}, count={count})")

    # Build figure
    fig = go.Figure()
    # Edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines', hoverinfo='none', showlegend=False,
        line=dict(color='gray', width=1)
    ))
    # Nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=sizes, color=base_color, line=dict(width=1, color='white')),
        text=tips,
        hoverinfo='text',
        showlegend=False
    ))
    # Node labels on plot
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='text',
        text=labels,
        textposition='middle center',
        textfont=dict(size=[ CENTER_FONT_SIZE ] + [PLOT_FONT_SIZE] * (len(labels)-1 )),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        title=f"Neighbors of '{token}'" + (f" ({year})" if year else ""),
        template='plotly_dark',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=1200,
        height=800,
        hoverlabel=dict(font=dict(size=HOVER_FONT_SIZE)),
        uirevision="constant"
    )

    return fig


# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Header(html.H1("EEBO Semantic Drift Dashboard")),

    html.Section([
        html.Label("View mode:"),
        dcc.RadioItems(
            id='normalize-toggle',
            options=[{"label":"Raw","value":"raw"},{"label":"Normalized","value":"norm"}],
            value="raw",
            inline=True
        ),
        dcc.Graph(id='main-dashboard', figure=create_main_dashboard_figure(data))
    ]),

    html.Section([
        html.Label(["Select token:",
            dcc.Dropdown(
                id='token-dropdown',
                options=[{"label":t,"value":t} for t in sorted(tokens)],
                value=tokens[0] if tokens else None
            ),
        ]),
        html.Label("Select year:"),
        dcc.Slider(
            id='year-slider',
            min=min(all_years),
            max=max(all_years),
            step=1,
            marks={y:str(y) for y in all_years},
            value=min(all_years)
        ),
        html.Button("Start/Stop Animation", id='anim-button'),
        dcc.Graph(id='neighbor-graph'),
    ]),

    dcc.Store(id='selected-point', data={'token': None, 'year': None}),
    dcc.Store(id='animation-state', data=False),
    dcc.Interval(id='animate-interval', interval=1000, n_intervals=0, disabled=True)
])

@app.callback(
    Output('selected-point', 'data'),
    Input('main-dashboard', 'clickData'),
    Input('token-dropdown', 'value'),
    Input('year-slider', 'value'),
    State('selected-point', 'data')
)
def update_selected_point(clickData, dropdown_value, slider_year, current):
    ctx = dash.callback_context

    if not current or current.get('token') is None:
        current = {
            'token': dropdown_value if dropdown_value else (tokens[0] if tokens else None),
            'year': slider_year if slider_year else (min(all_years) if all_years else None)
        }

    if not ctx.triggered:
        return current

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'main-dashboard' and clickData:
        point = clickData['points'][0]
        year = point.get('x')
        cd = point.get('customdata') or {}
        token = cd.get('token')
        color = cd.get('color')
        if token is None:
            return current
        return {'token': token, 'year': year if year is not None else current['year'], 'color': color}

    elif trigger == 'token-dropdown' and dropdown_value:
        return {'token': dropdown_value, 'year': current['year']}

    elif trigger == 'year-slider':
        return {'token': current['token'], 'year': slider_year}

    return current

@app.callback(
    Output('main-dashboard', 'figure'),
    Input('normalize-toggle', 'value'),
    Input('year-slider', 'value')
)
def update_main(normalize_mode, highlight_year):
    logger.info(f"[DASH update_main] {normalize_mode} year={highlight_year}")
    normalize = normalize_mode=="norm"
    return create_main_dashboard_figure(data, normalize=normalize, highlight_year=highlight_year)

@app.callback(
    Output('neighbor-graph', 'figure'),
    Input('selected-point', 'data'),
    Input('animate-interval', 'n_intervals'),
    State('animation-state', 'data')
)
def update_neighbors(selected_point, n_intervals, anim_state):
    if not selected_point or not selected_point.get("token"):
        return go.Figure()

    token = selected_point["token"]
    year = selected_point["year"]
    color = selected_point.get("color", "#222")

    if anim_state and n_intervals:
        i = all_years.index(year)
        year = all_years[(i+1) % len(all_years)]

    slices = data[token].get("slices", [])
    if not slices:
        return go.Figure()

    s = next((x for x in slices if x["year"] == year), slices[-1])
    neighbors = s.get("top_neighbors", [])


    neighbors = [
        {"token": n.get("token"), "similarity": n.get("similarity", 0), "count": n.get("count", -2)}
        for n in neighbors if isinstance(n, dict)
    ]

    logger.info(f"[DASH update_neighbours] neighbors={neighbors}")

    return create_neighbor_figure(token, neighbors, year, color)

@app.callback(
    Output('animation-state', 'data'),
    Output('animate-interval', 'disabled'),
    Input('anim-button', 'n_clicks'),
    Input('main-dashboard', 'clickData'),
    State('animation-state', 'data'),
    prevent_initial_call=True
)
def control_animation(n_clicks, clickData, state):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger=='main-dashboard':
        return False, True
    if trigger=='anim-button':
        return not state, state
    return state, not state

if __name__=="__main__":
    app.run(debug=True)
