#!/usr/bin/env python
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mb_test import OUT_PATH

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_interactive_dashboard(data):
    # Multi-metric subplot
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
        subplot_titles=["Semantic Drift and Phase Transitions"]
    )

    for token, token_data in data.items():
        slices = token_data.get("slices", [])
        if not slices:
            continue

        years = [s["year"] for s in slices]
        drift = [s.get("drift", 0) for s in slices]
        jsd = [s.get("js_divergence", 0) for s in slices]
        entropy = [s.get("entropy", 0) for s in slices]
        cluster_sizes = [s.get("cluster_sizes", []) for s in slices]
        births = [s.get("births", 0) for s in slices]
        top_neighbors = [", ".join(f"{w}:{c}" for w, c in s.get("top_neighbors", [])[:5]) for s in slices]

        # Drift line
        fig.add_trace(go.Scatter(
            x=years,
            y=drift,
            mode='lines+markers',
            name=f"{token} drift",
            line=dict(width=2),
            marker=dict(size=6),
            hovertemplate=(
                f"<b>{token}</b><br>"
                "Year: %{x}<br>"
                "Drift: %{y:.4f}<br>"
                "JSD: %{customdata[0]:.4f}<br>"
                "Entropy: %{customdata[1]:.4f}<br>"
                "Cluster sizes: %{customdata[2]}<br>"
                "Births: %{customdata[3]}<br>"
                "Top neighbors: %{customdata[4]}"
            ),
            customdata=list(zip(jsd, entropy, cluster_sizes, births, top_neighbors, strict=True)),
        ), secondary_y=False)

        # JSD line on secondary y-axis
        fig.add_trace(go.Scatter(
            x=years,
            y=jsd,
            mode='lines+markers',
            name=f"{token} JSD",
            line=dict(width=2, dash='dot'),
            marker=dict(symbol='triangle-up', size=6),
            hovertemplate="<b>" + token + "</b><br>Year: %{x}<br>JSD: %{y:.4f}"
        ), secondary_y=True)

        # Phase transition markers
        pt_data = token_data.get("phase_transitions", {})
        for t in pt_data.get("major", []):
            fig.add_vline(
                x=t["year"],
                line=dict(color='red', width=2, dash='dash'),
                annotation_text=f"{token} MAJOR @ {t['year']}",
                annotation_position="top right"
            )
        for t in pt_data.get("minor", []):
            fig.add_vline(
                x=t["year"],
                line=dict(color='orange', width=1, dash='dot'),
                annotation_text=f"{token} MINOR @ {t['year']}",
                annotation_position="top left"
            )
        for t in pt_data.get("single_doc_spikes", []):
            fig.add_vline(
                x=t["year"],
                line=dict(color='yellow', width=1, dash='dash'),
                annotation_text=f"{token} SINGLE-DOC @ {t['year']}",
                annotation_position="bottom right"
            )

    fig.update_layout(
        title="Semantic Drift, JS Divergence, and Phase Transitions",
        xaxis_title="Year",
        yaxis_title="Drift",
        yaxis2=dict(title="JS Divergence", overlaying='y', side='right'),
        hovermode="closest",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(font_size=14)
    )

    fig.show()


def main():
    data = load_data(OUT_PATH)
    create_interactive_dashboard(data)


if __name__ == "__main__":
    main()
