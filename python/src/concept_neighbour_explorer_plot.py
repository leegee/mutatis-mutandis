from typing import Dict, List
import json
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
import networkx as nx
import matplotlib.pyplot as plt

import lib.eebo_config as config
from lib.eebo_logging import logger

INPUT_FILE = config.OUT_DIR / "concept_neighbour_audit.json"
SVG_FILE = INPUT_FILE.with_suffix(".svg")
MIN_FREQ = 5
MIN_SIM = 0.5
GANTT_YEAR_SCALE = 0.3

@dataclass
class NeighbourStats:
    similarities: List[float] = field(default_factory=list)
    total_frequency: int = 0
    occurrences: int = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data: Dict[str, Dict[str, Dict]] = json.load(f)

summary_by_slice: Dict[str, List[Dict]] = {}
token_trajectories: Dict[str, List[Dict]] = defaultdict(list)

# Build summary per slice
for slice_name, probes in data.items():
    neighbour_data: defaultdict[str, NeighbourStats] = defaultdict(NeighbourStats)

    for _probe_name, probe_info in probes.items():
        neighbours = probe_info.get("neighbours", [])
        for n in neighbours:
            sim = float(n["similarity"])
            freq = int(n.get("frequency", 0))
            if freq < MIN_FREQ or sim < MIN_SIM:
                continue  # skip low-frequency or low-sim neighbours

            token: str = n["token"]
            neighbour_data[token].similarities.append(sim)
            neighbour_data[token].total_frequency += freq
            neighbour_data[token].occurrences += 1

    # build slice summary and assign rank by frequency
    slice_summary: List[Dict] = []
    for token, stats in neighbour_data.items():
        slice_summary.append({
            "token": token,
            "avg_similarity": mean(stats.similarities),
            "total_frequency": stats.total_frequency,
            "times_as_neighbour": stats.occurrences
        })

    # sort by total_frequency descending to assign ranks
    slice_summary.sort(key=lambda x: x["total_frequency"], reverse=True)
    for rank, entry in enumerate(slice_summary, start=1):
        token_trajectories[entry["token"]].append({
            "slice": slice_name,
            "rank": rank,
            "frequency": entry["total_frequency"],
            "avg_similarity": entry["avg_similarity"]
        })

    summary_by_slice[slice_name] = slice_summary

# Log summary
for slice_name, neighbours in summary_by_slice.items():
    logger.info(f"\n## SLICE {slice_name} ##")
    for n in neighbours:
        if n["total_frequency"] >= MIN_FREQ:
            logger.info(
                f"{n['token']:15} "
                f"freq={n['total_frequency']:5} "
                f"avg_sim={n['avg_similarity']:.3f} "
                f"seen={n['times_as_neighbour']}"
            )

# Plot trajectories
plt.figure(figsize=(12, 6))

for token, trajectory in token_trajectories.items():
    xs: List[int] = [
        int(entry["slice"].split("_")[0]) +
        (int(entry["slice"].split("_")[1]) - int(entry["slice"].split("_")[0])) // 2
        for entry in trajectory
    ]
    ys: List[int] = [entry["rank"] for entry in trajectory]

    # sort by xs to ensure proper plotting order
    sorted_pairs = sorted(zip(xs, ys, strict=True), key=lambda p: p[0])
    xs_sorted, ys_sorted = map(list, zip(*sorted_pairs, strict=True))

    plt.plot(xs_sorted, ys_sorted, marker="o", label=token)

plt.gca().invert_yaxis()  # rank 1 at top
plt.xlabel("Year (mid-slice)")
plt.ylabel("Neighbour rank (1 = highest frequency)")
plt.title("Neighbour-rank trajectories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






# Plot network

G = nx.DiGraph()

# Assign a column index for each slice (sorted)
slice_names = sorted(summary_by_slice.keys())
slice_to_col = {s: i for i, s in enumerate(slice_names)}

for token, trajectory in token_trajectories.items():
    if all(entry["frequency"] >= MIN_FREQ for entry in trajectory):
        cols = [slice_to_col[entry["slice"]] for entry in trajectory]
        ranks = [entry["rank"] for entry in trajectory]

        # add nodes and edges
        for i in range(len(trajectory)):
            node_id = f"{token}_{cols[i]}"
            G.add_node(node_id, label=token, col=cols[i], rank=ranks[i], slice=trajectory[i]["slice"])
            if i > 0:
                prev_node_id = f"{token}_{cols[i-1]}"
                G.add_edge(prev_node_id, node_id)

# Determine max label length for spacing
max_label_len = max(len(n.split("_")[0]) for n in G.nodes())


# Position nodes

# Build mapping from column index to nodes in that column
col_to_nodes = defaultdict(list)
for node_id, attrs in G.nodes(data=True):
    col_to_nodes[attrs["col"]].append(node_id)  # attrs["col"] must be numeric

pos = {}
for col, nodes_in_col in col_to_nodes.items():  # col is numeric
    n = len(nodes_in_col)
    for i, node_id in enumerate(nodes_in_col):
        offset = (i - (n - 1) / 2) * max_label_len * 0.25
        pos[node_id] = (col + offset, -float(G.nodes[node_id]["rank"]))


# Plot

plt.figure(figsize=(max(12, len(slice_names) * 1.5), 6))  # wide figure for many slices

nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=600,
    node_color="#ddd", # invisible
    edgecolors="#555", # invisible
)

# Add token labels above nodes
for node_id, (x, y) in pos.items():
    plt.text(
        x,
        y , # over node
        G.nodes[node_id]["label"],
        fontsize=9,
        ha="center",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", pad=1)  # makes label more readable
    )

plt.xlabel("Year slices")
plt.ylabel("Neighbour rank (1 = highest frequency)")
plt.title("Neighbour trajectories as network nodes")
plt.grid(True)
plt.tight_layout()
plt.show()







# Gantt chart


# Prepare Gantt chart data
slice_names = sorted(summary_by_slice.keys())
slice_starts = [int(s.split("_")[0]) for s in slice_names]
slice_ends = [int(s.split("_")[1]) for s in slice_names]
slice_widths = [end - start + 1 for start, end in zip(slice_starts, slice_ends, strict=True)]

tokens = sorted(token_trajectories.keys())
num_tokens = len(tokens)

# Build a mapping: token -> list of (start_year, width, is_present)
gantt_data: Dict[str, List[tuple]] = defaultdict(list)

for token in tokens:
    trajectory_slices = {entry["slice"]: entry for entry in token_trajectories[token]}
    for slice_name, start, width in zip(slice_names, slice_starts, slice_widths, strict=True):
        if slice_name in trajectory_slices and trajectory_slices[slice_name]["frequency"] >= MIN_FREQ:
            gantt_data[token].append((start, width, True))  # present
        else:
            gantt_data[token].append((start, width, False))  # absent


# Plot Gantt chart
fig, ax = plt.subplots(figsize=(max(24, len(slice_names) * 0.6), max(0.5*num_tokens, 6)))

bar_height = 0.8

for i, token in enumerate(tokens):
    for start, width, present in gantt_data[token]:
        scaled_start = start * GANTT_YEAR_SCALE
        scaled_width = width * GANTT_YEAR_SCALE
        if present:
            ax.broken_barh(
                [(scaled_start, scaled_width)],
                (i - bar_height/2, bar_height),
                facecolors='skyblue', edgecolors='black'
            )
        else:
            # draw a thin gray line for missing slices
            ax.broken_barh(
                [(scaled_start, scaled_width)],
                (i - bar_height/2, bar_height),
                facecolors='none', edgecolors='lightgray', linestyle='dashed'
            )

# Y-axis labels = tokens
ax.set_yticks(range(num_tokens))
ax.set_yticklabels(tokens, fontsize=12)

# X-axis = years
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Tokens", fontsize=14)
ax.set_title("EEBO token trajectories in non-Latin shorter texts", fontsize=16)
xticks = [int(s.split("_")[0]) for s in slice_names]
ax.set_xticks([x * GANTT_YEAR_SCALE for x in xticks])
ax.set_xticklabels([str(x) for x in xticks], fontsize=12)

# grid for clarity
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(SVG_FILE, format='svg')
logger.info(f"Gantt chart SVG written to {SVG_FILE}")

plt.show()
