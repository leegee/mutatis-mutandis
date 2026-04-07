#!/usr/bin/env python
import json
import plotly.graph_objects as go
from mb_test import OUT_PATH

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_interactive_plot(data):
    fig = go.Figure()

    for token, token_data in data.items():
        slices = token_data.get("slices", [])
        if not slices:
            continue

        years = [s["year"] for s in slices]
        drift = [s.get("drift", 0) for s in slices]
        jsd = [s.get("js_divergence", 0) for s in slices]
        births = [s.get("births", 0) for s in slices]
        top_neighbors = [", ".join(f"{w}:{c}" for w, c in s.get("top_neighbors", [])[:5]) for s in slices]

        # Drift line
        fig.add_trace(go.Scatter(
            x=years,
            y=drift,
            mode='lines+markers',
            name=f"{token} drift",
            line=dict(width=2),
            hovertemplate=(
                f"<b>{token}</b><br>"
                "Year: %{x}<br>"
                "Drift: %{y:.4f}<br>"
                "JSD: %{customdata[0]:.4f}<br>"
                "Births: %{customdata[1]}<br>"
                "Top neighbors: %{customdata[2]}"
            ),
            customdata=list(zip(jsd, births, top_neighbors, strict=True))
        ))

        # Phase transitions markers
        pt_data = token_data.get("phase_transitions", {}).get("major", [])
        for pt in pt_data:
            fig.add_vline(
                x=pt["year"],
                line=dict(color='red', width=2, dash='dash'),
                annotation_text=f"{token} MAJOR PHASE @ {pt['year']}",
                annotation_position="top right"
            )

    fig.update_layout(
        title="Semantic Drift, JSD, and Phase Transitions",
        xaxis_title="Year",
        yaxis_title="Drift",
        hovermode="closest",
        template="plotly_dark",
        hoverlabel=dict(
            font_size=16,
        )
    )

    fig.show()

def main():
    data = load_data(OUT_PATH)
    create_interactive_plot(data)

if __name__ == "__main__":
    main()
