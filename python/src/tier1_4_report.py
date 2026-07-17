#!/usr/bin/env python
"""
tier1_4_report.py - Reporting & visualization for the Tier 1.4 substitute-profile store.

Consumes the artifacts written by tier1_4_substitute_vectors.py:

    accumulator.pkl        (loaded via SubstituteVectorStore: corpus-wide
                             AND per-year profiles)
    substitute_matrix.npz  (corpus-wide, type x vocab, CSR)
    types.json
    type_counts.json
    year_counts.json / .csv
    vocab_strings.json     (vocab id -> surface string; requires a
                             tier1_4_substitute_vectors.py run that
                             includes export_vocab_strings)

and produces two genuinely different kinds of report:

  1. SYNONYM CANDIDATES (corpus-wide, cross-sectional)
     `synonyms` / `network` subcommands. Ranks/graphs words by
     Jensen-Shannon divergence between their aggregate substitute
     profiles. This is what Tier 1.4 is directly good at: "what does
     the model treat as interchangeable with this word, averaged over
     every occurrence in the corpus."

  2. DIACHRONIC DRIFT (per-year, using the per-year accumulator)
     `drift` / `entropy` subcommands. Tracks how a word's substitute
     profile moves across 1625-1689, with Civil War / Interregnum /
     Restoration period shading, and reliability (occurrence count)
     encoded directly in point size/style so sparse years don't look
     as confident as well-attested ones.

IMPORTANT LIMITATION - this is NOT sense clustering / polysemy detection
--------------------------------------------------------------------------
Tier 1.4's profile for a wordform is a SINGLE distribution, summed
across every occurrence of that word regardless of which sense was in
play at that occurrence. A genuinely polysemous word - one sense
inherited from an earlier period, a second sense emerging later - does
NOT show up here as two separable clusters. It shows up as one blended,
averaged profile; evidence of the second sense is smeared into the
tail of the distribution rather than visible as a distinct mode.

The `entropy` subcommand is offered as a weak, indirect PROXY for
distributional breadth over time: a word whose per-year substitute
distribution flattens out (entropy rising) is *consistent with*
acquiring additional usages, but entropy alone can't distinguish
"gained a new sense" from "became vaguer" from "just got noisier from
a smaller sample that year" - always read it next to the occurrence
count, not on its own.

Actually detecting distinct senses (and when one splits from another)
needs INSTANCE-level data: clustering the individual per-occurrence
embeddings Tier 1 stores (one hidden-state vector per masked occurrence)
rather than the pre-aggregated bag-of-substitutes Tier 1.4 stores.
That's a natural follow-up tier (cluster Tier 1's per-occurrence
vectors for a wordform, e.g. via HDBSCAN, and look for >1 stable
cluster per period) but is out of scope here - this script only ever
reports on what Tier 1.4 actually contains.

Usage
-----
    python tier1_4_report.py synonyms liberty
    python tier1_4_report.py drift liberty --baseline first
    python tier1_4_report.py entropy liberty
    python tier1_4_report.py sparsity --words liberty,tyranny,freedome
    python tier1_4_report.py network --threshold 0.45

All figures are written as PNGs into the Tier 1.4 store directory
(override with --path). `network` additionally writes a .gexf for
Gephi / further exploration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tier1_4_substitute_vectors import SubstituteVectorStore, TIER1_4_PATH


# Rough period shading for the Civil War / Interregnum / Restoration
# window. These are the usual textbook approximations, not a claim of
# precision - adjust boundaries to your own periodisation as needed.
PERIODS = [
    ("Personal Rule", 1625, 1640, "#dbe7f5"),
    ("Civil Wars",    1642, 1651, "#f5d6d6"),
    ("Interregnum",   1649, 1660, "#e8dbf5"),
    ("Restoration",   1660, 1689, "#dff5e0"),
]


def _shade_periods(ax, year_min: int, year_max: int) -> None:
    for label, start, end, color in PERIODS:
        s, e = max(start, year_min), min(end, year_max)
        if s < e:
            ax.axvspan(s, e, color=color, alpha=0.5, zorder=0)
    handles = [Patch(facecolor=c, alpha=0.5, label=l) for l, s, e, c in PERIODS]
    ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)


def _load_json(path: Path, name: str):
    fp = path / name
    if not fp.exists():
        raise SystemExit(f"{fp} not found - run tier1_4_substitute_vectors.py "
                          f"(without --export-only skipped) first.")
    with open(fp) as f:
        return json.load(f)


def load_vocab(path: Path) -> list[str]:
    fp = path / "vocab_strings.json"
    if not fp.exists():
        raise SystemExit(
            f"{fp} not found. Re-run tier1_4_substitute_vectors.py - it now calls "
            f"store.export_vocab_strings(tokenizer) in main(), which writes this file."
        )
    with open(fp) as f:
        return json.load(f)


def decode_dist(dist: dict[int, float], vocab: list[str], top_n: int = 10):
    """Turn a {vocab_id: prob} dict into a sorted [(word, prob), ...] list."""
    items = sorted(dist.items(), key=lambda kv: -kv[1])[:top_n]
    return [(vocab[vid] if vid < len(vocab) else f"<id:{vid}>", p) for vid, p in items]


# ---------------------------------------------------------------- synonyms

def cmd_synonyms(args):
    from scipy.spatial.distance import jensenshannon

    matrix = sp.load_npz(args.path / "substitute_matrix.npz")
    types = _load_json(args.path, "types.json")
    type_counts = _load_json(args.path, "type_counts.json")
    vocab = load_vocab(args.path)

    query = args.word.strip().lower()
    if query not in types:
        raise SystemExit(f"{query!r} not in exported types "
                          f"(below --min-count when exported, or not in CONCEPT_SETS at all)")

    idx = types.index(query)
    qvec = np.asarray(matrix[idx].todense()).ravel()
    qvec = qvec / (qvec.sum() + 1e-12)

    scores = []
    for i, w in enumerate(types):
        if w == query:
            continue
        row = np.asarray(matrix[i].todense()).ravel()
        s = row.sum()
        if s == 0:
            continue
        row = row / s
        d = jensenshannon(qvec, row)
        if np.isnan(d):
            continue
        scores.append((w, d, type_counts.get(w, 0)))
    scores.sort(key=lambda x: x[1])
    top = scores[: args.top_n]

    if not top:
        raise SystemExit(f"No comparable types found for {query!r}.")

    print(f"\nNearest substitutes for {query!r} "
          f"(corpus-wide, n={type_counts.get(query, 0)}):")
    for w, d, c in top:
        print(f"  {w:20s}  JS={d:.4f}  n={c}")

    words = [w for w, d, c in top][::-1]
    sims = [1 - d for w, d, c in top][::-1]
    counts = [c for w, d, c in top][::-1]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(words))))
    bars = ax.barh(words, sims, color="#4c72b0")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"n={c}", va="center", fontsize=8, color="#555")
    ax.set_xlabel("Substitute-profile similarity (1 - JS divergence)")
    ax.set_title(f'Nearest substitutes for "{query}" (corpus-wide, {args.year_min}-{args.year_max})')
    ax.set_xlim(0, 1)
    fig.tight_layout()
    out = args.path / f"synonyms_{query}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Also show what the model itself predicts fills the query word's
    # own slot - context, NOT a synonym ranking (dominated by the word's
    # own id per the "own vocabulary id dominates" limitation), but
    # useful as a sanity check that the masked-LM signal looks sane.
    store = SubstituteVectorStore(args.path)
    raw = store.agg.get(query, {})
    count = store.type_counts.get(query, 0)
    if raw and count:
        normalized = {vid: p / count for vid, p in raw.items()}
        decoded = decode_dist(normalized, vocab, top_n=10)
        print(f"\nModel's own top fill-in predictions for {query!r} (sanity check, not synonymy):")
        for w, p in decoded:
            print(f"  {w:20s}  p={p:.3f}")


# ------------------------------------------------------------------ drift

def cmd_drift(args):
    from scipy.spatial.distance import jensenshannon

    store = SubstituteVectorStore(args.path)
    word = args.word.strip().lower()
    if store.type_counts.get(word, 0) == 0:
        raise SystemExit(f"{word!r} has no recorded occurrences in this store")

    years = list(range(args.year_min, args.year_max + 1))

    if args.baseline == "corpus":
        base_count = store.type_counts[word]
        base_dist = {vid: p / base_count for vid, p in store.agg[word].items()}
        base_label = "corpus-wide baseline"
    else:  # "first" attested year with enough data
        base_dist, base_label = None, None
        for y in years:
            dist, count, w = store.pooled_profile_auto(
                word, y, min_count=args.min_count, max_window=args.max_window)
            if count >= args.min_count:
                base_dist, base_label = dist, f"{y} (±{w}yr pool)"
                break
        if base_dist is None:
            raise SystemExit(
                f"{word!r} never reaches --min-count={args.min_count} in "
                f"{args.year_min}-{args.year_max}, even pooled up to ±{args.max_window} "
                f"years. Try --baseline corpus, or lower --min-count."
            )

    plot_years, divs, counts, windows = [], [], [], []
    for y in years:
        dist, count, w = store.pooled_profile_auto(
            word, y, min_count=args.min_count, max_window=args.max_window)
        if count == 0:
            continue
        keys = set(base_dist) | set(dist)
        bvec = np.array([base_dist.get(k, 0.0) for k in keys])
        dvec = np.array([dist.get(k, 0.0) for k in keys])
        bvec = bvec / (bvec.sum() + 1e-12)
        dvec = dvec / (dvec.sum() + 1e-12)
        d = jensenshannon(bvec, dvec)
        if np.isnan(d):
            continue
        plot_years.append(y)
        divs.append(d)
        counts.append(count)
        windows.append(w)

    if not plot_years:
        raise SystemExit(f"No year had enough data for {word!r} even after pooling.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    _shade_periods(ax, args.year_min, args.year_max)

    sizes = [20 + 6 * c for c in counts]
    exact = [i for i, w in enumerate(windows) if w == 0]
    pooled = [i for i, w in enumerate(windows) if w > 0]

    ax.plot(plot_years, divs, "--", color="#888", linewidth=1, zorder=1)
    if exact:
        ax.scatter([plot_years[i] for i in exact], [divs[i] for i in exact],
                    s=[sizes[i] for i in exact], color="#c44e52", zorder=2, label="exact year")
    if pooled:
        ax.scatter([plot_years[i] for i in pooled], [divs[i] for i in pooled],
                    s=[sizes[i] for i in pooled], facecolors="none",
                    edgecolors="#c44e52", linewidths=1.5, zorder=2, label="pooled (sparse year)")

    ax.set_xlabel("Year")
    ax.set_ylabel(f"JS divergence from {base_label}")
    ax.set_title(f'Substitute-profile drift: "{word}" ({args.year_min}-{args.year_max})')
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = args.path / f"drift_{word}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


# --------------------------------------------------------------- entropy

def cmd_entropy(args):
    store = SubstituteVectorStore(args.path)
    word = args.word.strip().lower()
    if store.type_counts.get(word, 0) == 0:
        raise SystemExit(f"{word!r} has no recorded occurrences in this store")

    years = list(range(args.year_min, args.year_max + 1))

    plot_years, ents, counts, windows = [], [], [], []
    for y in years:
        dist, count, w = store.pooled_profile_auto(
            word, y, min_count=args.min_count, max_window=args.max_window)
        if count == 0:
            continue
        probs = np.array([p for p in dist.values() if p > 0])
        ent = float(-(probs * np.log2(probs)).sum())
        plot_years.append(y)
        ents.append(ent)
        counts.append(count)
        windows.append(w)

    if not plot_years:
        raise SystemExit(f"No year had enough data for {word!r}.")

    fig, ax = plt.subplots(figsize=(9, 4))
    _shade_periods(ax, args.year_min, args.year_max)
    sizes = [20 + 6 * c for c in counts]
    exact = [i for i, w in enumerate(windows) if w == 0]
    pooled = [i for i, w in enumerate(windows) if w > 0]

    ax.plot(plot_years, ents, "--", color="#888", linewidth=1, zorder=1)
    if exact:
        ax.scatter([plot_years[i] for i in exact], [ents[i] for i in exact],
                    s=[sizes[i] for i in exact], color="#4c72b0", zorder=2, label="exact year")
    if pooled:
        ax.scatter([plot_years[i] for i in pooled], [ents[i] for i in pooled],
                    s=[sizes[i] for i in pooled], facecolors="none",
                    edgecolors="#4c72b0", linewidths=1.5, zorder=2, label="pooled (sparse year)")

    ax.set_xlabel("Year")
    ax.set_ylabel("Substitute-distribution entropy (bits)")
    ax.set_title(f'Distributional breadth over time: "{word}"\n'
                  f'(weak polysemy/vagueness proxy - not a sense count, see script docstring)',
                  fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = args.path / f"entropy_{word}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


# -------------------------------------------------------------- sparsity

def cmd_sparsity(args):
    table = _load_json(args.path, "year_counts.json")

    if args.words:
        words = [w.strip().lower() for w in args.words.split(",")]
        missing = [w for w in words if w not in table]
        if missing:
            raise SystemExit(f"Not in year_counts.json: {missing}")
    else:
        # default: the --top-n words with the highest corpus-wide total
        totals = {w: sum(v for v in yc.values()) for w, yc in table.items()}
        words = sorted(totals, key=lambda w: -totals[w])[: args.top_n]

    years = sorted({int(y) for w in words for y in table[w].keys()})
    grid = np.array([[table[w][str(y)] for y in years] for w in words])

    fig, ax = plt.subplots(figsize=(0.25 * len(years) + 2, 0.35 * len(words) + 2))
    im = ax.imshow(np.log1p(grid), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=90, fontsize=6)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("log(1 + occurrence count)")
    ax.set_title("Tier 1.4 per-year coverage")
    fig.tight_layout()
    out = args.path / "sparsity_heatmap.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


# --------------------------------------------------------------- network

def cmd_network(args):
    from scipy.spatial.distance import jensenshannon
    try:
        import networkx as nx
    except ImportError:
        raise SystemExit("networkx is required: pip install networkx --break-system-packages")

    matrix = sp.load_npz(args.path / "substitute_matrix.npz")
    types = _load_json(args.path, "types.json")
    type_counts = _load_json(args.path, "type_counts.json")

    dense = np.asarray(matrix.todense())
    sums = dense.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    norm = dense / sums

    G = nx.Graph()
    for w in types:
        G.add_node(w, count=type_counts.get(w, 0))

    n = len(types)
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(norm[i], norm[j])
            if np.isnan(d) or d > args.threshold:
                continue
            G.add_edge(types[i], types[j], weight=1 - d)

    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        raise SystemExit(f"No pairs below --threshold={args.threshold}; try raising it.")

    fig, ax = plt.subplots(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=0, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)))
    sizes = [80 + 4 * G.nodes[w]["count"] for w in G.nodes]
    widths = [G.edges[e]["weight"] * 3 for e in G.edges]
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4c72b0", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, edge_color="#999", alpha=0.6, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    ax.set_title(f"Synonym-candidate network (JS divergence < {args.threshold})")
    ax.axis("off")
    fig.tight_layout()
    out = args.path / "synonym_network.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    gexf_out = args.path / "synonym_network.gexf"
    nx.write_gexf(G, gexf_out)
    print(f"Also wrote {gexf_out} (Gephi / further exploration)")


# ------------------------------------------------------------------- CLI

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", type=Path, default=TIER1_4_PATH, help="Tier 1.4 store directory")
    sub = p.add_subparsers(dest="command", required=True)

    sp1 = sub.add_parser("synonyms", help="Corpus-wide nearest-substitute ranking for one word")
    sp1.add_argument("word")
    sp1.add_argument("--top-n", type=int, default=15)
    sp1.add_argument("--year-min", type=int, default=1625, help="Cosmetic, for the chart title only")
    sp1.add_argument("--year-max", type=int, default=1689, help="Cosmetic, for the chart title only")
    sp1.set_defaults(func=cmd_synonyms)

    sp2 = sub.add_parser("drift", help="Diachronic drift timeline for one word")
    sp2.add_argument("word")
    sp2.add_argument("--year-min", type=int, default=1625)
    sp2.add_argument("--year-max", type=int, default=1689)
    sp2.add_argument("--min-count", type=int, default=10)
    sp2.add_argument("--max-window", type=int, default=5)
    sp2.add_argument("--baseline", choices=["first", "corpus"], default="first")
    sp2.set_defaults(func=cmd_drift)

    sp3 = sub.add_parser("entropy", help="Distributional-breadth (weak polysemy proxy) timeline")
    sp3.add_argument("word")
    sp3.add_argument("--year-min", type=int, default=1625)
    sp3.add_argument("--year-max", type=int, default=1689)
    sp3.add_argument("--min-count", type=int, default=10)
    sp3.add_argument("--max-window", type=int, default=5)
    sp3.set_defaults(func=cmd_entropy)

    sp4 = sub.add_parser("sparsity", help="Word x year coverage heatmap")
    sp4.add_argument("--words", type=str, default=None,
                      help="Comma-separated wordforms; default = top --top-n by corpus-wide count")
    sp4.add_argument("--top-n", type=int, default=30)
    sp4.set_defaults(func=cmd_sparsity)

    sp5 = sub.add_parser("network", help="Synonym-candidate network graph across all types")
    sp5.add_argument("--threshold", type=float, default=0.5, help="Max JS divergence for an edge")
    sp5.set_defaults(func=cmd_network)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()