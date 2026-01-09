#!/usr/bin/env python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import eebo_config as config
import eebo_db

QUERY = "liberty"


df = pd.read_sql_query(f"""
    SELECT slice_start, slice_end, neighbour, rank, cosine
    FROM neighbourhoods
    WHERE query = '{QUERY}'
""", eebo_db.dbh)

eebo_db.dbh.close()

df['slice_label'] = df['slice_start'].astype(str)


# Heatmap of cosine similarity
df_top = df[df['rank'] <= config.TOP_K]
heatmap_data = df_top.pivot_table(
    index='slice_label',
    columns='neighbour',
    values='cosine'
).fillna(0)

plt.figure(figsize=(14, 6))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title(f"Cosine similarity of top '{QUERY}' neighbours over time")
plt.xlabel("Neighbour")
plt.ylabel("Slice / Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Rank vs cosine scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_top,
    x='rank',
    y='cosine',
    hue='slice_label',
    palette='viridis',
    s=100
)
plt.title(f"Rank vs Cosine similarity for '{QUERY}' neighbours")
plt.xlabel("Rank")
plt.ylabel("Cosine similarity")
plt.legend(title="Slice / Year", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Orthographic variant stacked bar
# Assumes anything that differs only by last few letters is a variant
# For simplicity, just show counts per neighbour per slice
variant_counts = df_top.groupby([
    'slice_label',
    'neighbour']
).size().unstack(fill_value=0)

variant_counts.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 6),
    colormap='tab20'
)
plt.title(f"Top '{QUERY}' orthographic variants over time")
plt.xlabel("Slice / Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Semantic network
G = nx.Graph()

for _, row in df_top.iterrows():
    G.add_node(row['neighbour'])
    G.add_edge(QUERY, row['neighbour'], weight=row['cosine'])

plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, k=0.5)
edges = G.edges(data=True)
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=1000,
    node_color='skyblue',
    font_size=10
)
nx.draw_networkx_edges(G, pos, width=[d['weight']*3 for (_, _, d) in edges])
plt.title(f"Semantic network of '{QUERY}' neighbours")
plt.show()


# Semantic drift line chart
# Uses mean cosine of top-K neighbours per slice as a proxy for stability
drift = df_top.groupby('slice_label')['cosine'].mean()
plt.figure(figsize=(12, 5))
drift.plot(marker='o')
plt.title(
    f"Average cosine of top '{QUERY}' neighbours over time (semantic drift proxy)"
)
plt.xlabel("Slice / Year")
plt.ylabel("Mean cosine similarity")
plt.grid(True)
plt.tight_layout()
plt.show()
