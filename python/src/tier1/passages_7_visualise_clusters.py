"""
passages_7_visualise_clusters.py

Turn the output of cluster_substitute_windows.py into a single self-contained
HTML page (no dependencies, works offline):

  * UMAP scatter of every window. Colour = cluster or term (toggle); shape = term.
  * Hover a point to read the passage (hit word highlighted), year, title,
    document and the selected collocates it contains.
  * Click legend entries to hide/show clusters or terms.
  * Click a cluster card to focus that cluster; each card shows size, documents,
    median year, distinctive collocates and a term-composition bar.

Run from the repository root:
    python src/tier1/visualise_clusters.py \
        --input out/resist_matched_profiles_clusters_windows.csv --open

By default the summary CSV (…_clusters_summary.csv) is found next to the input
and the HTML is written beside it as …_clusters.html.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import webbrowser
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def build_data(rows, summary_rows):
    needed = ("umap_x", "umap_y", "cluster", "term", "passage")
    missing = [c for c in needed if c not in rows[0]]
    if missing:
        sys.exit(f"Input is missing columns {missing}; use the "
                 f"…_clusters_windows.csv written by cluster_substitute_windows.py")

    terms: list[str] = []
    for r in rows:
        if r["term"] not in terms:
            terms.append(r["term"])
    term_idx = {t: i for i, t in enumerate(terms)}

    points = []
    for r in rows:
        points.append({
            "x": to_float(r["umap_x"]),
            "y": to_float(r["umap_y"]),
            "c": to_int(r["cluster"], -1),
            "t": term_idx[r["term"]],
            "yr": to_int(r.get("year")),
            "ti": r.get("title", ""),
            "d": r.get("doc_id", ""),
            "co": r.get("collocates_present", ""),
            "p": r["passage"],
            "pr": to_float(r.get("cluster_prob"), 1.0),
        })

    clusters = []
    if summary_rows:
        for s in summary_rows:
            c = -1 if s["cluster"] == "noise" else to_int(s["cluster"], -1)
            clusters.append({
                "cluster": c,
                "n": to_int(s.get("n_windows"), 0),
                "docs": to_int(s.get("n_docs"), 0),
                "yr": s.get("year_median", ""),
                "dc": s.get("distinctive_collocates", ""),
                "tc": s.get("term_counts", ""),
            })
    else:
        # Fallback: rebuild a minimal summary from the points themselves.
        for c in sorted({p["c"] for p in points}):
            pts = [p for p in points if p["c"] == c]
            years = [p["yr"] for p in pts if p["yr"] is not None]
            counts = Counter(terms[p["t"]] for p in pts)
            clusters.append({
                "cluster": c,
                "n": len(pts),
                "docs": len({p["d"] for p in pts}),
                "yr": int(statistics.median(years)) if years else "",
                "dc": "",
                "tc": ";".join(f"{t}:{n}" for t, n in counts.items()),
            })
    return {"points": points, "terms": terms, "clusters": clusters}


def main():
    ap = argparse.ArgumentParser(description="Visualise cluster output as HTML.")
    ap.add_argument("--input", type=Path, required=True,
                    help="…_clusters_windows.csv from cluster_substitute_windows.py")
    ap.add_argument("--summary", type=Path, default=None,
                    help="…_clusters_summary.csv (default: found beside the input).")
    ap.add_argument("--output", type=Path, default=None,
                    help="HTML path (default: input stem without '_windows' + .html).")
    ap.add_argument("--title", default=None)
    ap.add_argument("--open", action="store_true", help="Open the page in a browser.")
    args = ap.parse_args()

    rows = read_csv(args.input)
    if not rows:
        sys.exit(f"No rows in {args.input}")

    stem = args.input.stem
    base = stem[: -len("_windows")] if stem.endswith("_windows") else stem
    summary_path = args.summary or args.input.with_name(base + "_summary.csv")
    summary_rows = read_csv(summary_path) if summary_path.exists() else []
    if not summary_rows:
        print(f"Note: no summary CSV at {summary_path}; rebuilding a minimal one.")

    data = build_data(rows, summary_rows)
    title = args.title or f"Window clusters: {', '.join(data['terms'][:3])}" + (
        "…" if len(data["terms"]) > 3 else "")
    payload = json.dumps(data).replace("</", "<\\/")
    html = HTML.replace("__TITLE__", title).replace("__DATA__", payload)

    out = args.output or args.input.with_name(base + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} windows, "
          f"{len([c for c in data['clusters'] if c['cluster'] >= 0])} clusters)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#fff; --fg:#1a1a1a; --muted:#666; --card:#f5f5f2; --line:#d8d8d4; --sel:#1a1a1a; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161616; --fg:#eee; --muted:#9a9a9a; --card:#222; --line:#444; --sel:#eee; }
  }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
         background:var(--bg); color:var(--fg); }
  #app { display:flex; gap:16px; padding:16px; height:100vh; }
  #plot { flex:1 1 60%; min-width:0; display:flex; flex-direction:column; gap:8px; }
  #stage { flex:1; min-height:0; }
  canvas { width:100%; height:100%; border:1px solid var(--line); border-radius:6px; display:block; }
  #side { flex:0 0 400px; overflow:auto; padding-right:4px; }
  h1 { font-size:16px; margin:0; }
  .sub { color:var(--muted); font-size:12px; }
  .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; font-size:13px; }
  .controls label { display:flex; gap:4px; align-items:center; cursor:pointer; }
  button { font:inherit; background:var(--card); color:var(--fg); border:1px solid var(--line);
           border-radius:4px; padding:2px 8px; cursor:pointer; }
  .legend { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { display:inline-flex; align-items:center; gap:5px; padding:2px 8px; border-radius:12px;
          border:1px solid var(--line); background:var(--card); cursor:pointer; font-size:12px; user-select:none; }
  .chip.off { opacity:.35; text-decoration:line-through; }
  .sw { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:8px 10px;
          margin-bottom:8px; cursor:pointer; }
  .card.sel { outline:2px solid var(--sel); }
  .card h3 { margin:0 0 2px; font-size:13px; display:flex; gap:6px; align-items:center; }
  .tags { display:flex; flex-wrap:wrap; gap:4px; margin:4px 0; }
  .tag { font-size:11px; padding:1px 6px; border-radius:3px; background:var(--bg); border:1px solid var(--line); }
  .bar { display:flex; height:10px; border-radius:3px; overflow:hidden; margin-top:4px; background:var(--line); }
  .bar div { height:100%; }
  .bartxt { font-size:11px; color:var(--muted); margin-top:2px; }
  #tip { position:fixed; display:none; max-width:440px; padding:8px 10px; pointer-events:none; z-index:10;
         background:var(--bg); color:var(--fg); border:1px solid var(--line); border-radius:6px;
         box-shadow:0 4px 14px rgba(0,0,0,.25); font-size:12px; }
  #tip .meta { color:var(--muted); margin-bottom:4px; }
  #tip mark { background:#ffd54a; color:#111; padding:0 2px; border-radius:2px; font-weight:600; }
  @media (max-width: 900px) { #app { flex-direction:column; height:auto; } #stage { height:60vh; flex:none; }
    #side { flex:none; } }
</style>
</head>
<body>
<div id="app">
  <div id="plot">
    <div><h1>__TITLE__</h1><div class="sub" id="counts"></div></div>
    <div class="controls">
      <span>Colour by</span>
      <label><input type="radio" name="mode" value="cluster" checked> cluster</label>
      <label><input type="radio" name="mode" value="term"> term</label>
      <label>Point size <input type="range" id="size" min="2" max="9" value="4"></label>
      <button id="reset">Reset</button>
    </div>
    <div class="legend" id="colorLegend"></div>
    <div class="legend" id="shapeLegend"></div>
    <div id="stage"><canvas id="cv"></canvas></div>
  </div>
  <div id="side"><div id="cards"></div></div>
</div>
<div id="tip"></div>

<script>
(function () {
  const D = __DATA__;
  const pts = D.points, terms = D.terms, summary = D.clusters;
  const PAL = ["#4e79a7","#f28e2b","#e15759","#76b7b2","#59a14f","#edc948","#b07aa1","#ff9da7","#9c755f","#17becf"];
  const NOISE = "#999999";
  const SHAPES = ["circle","square","triangle","diamond","plus","cross"];
  const GLYPH = { circle:"●", square:"■", triangle:"▲", diamond:"◆", plus:"✚", cross:"✖" };

  const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
  const tip = document.getElementById("tip"), sizeEl = document.getElementById("size");
  let mode = "cluster", focus = null, hover = -1, W = 0, H = 0, dpr = 1;
  const hiddenColor = new Set(), hiddenTerm = new Set();

  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);

  const colorOf = p => mode === "term" ? PAL[p.t % PAL.length] : (p.c < 0 ? NOISE : PAL[p.c % PAL.length]);
  const colorKey = p => mode === "term" ? "t" + p.t : "c" + p.c;
  const shapeOf = p => SHAPES[p.t % SHAPES.length];
  const visible = p => !hiddenColor.has(colorKey(p)) && !hiddenTerm.has(p.t);

  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function renderPassage(parent, passage) {
    const re = /\[\[(.+?)\]\]/g;
    let last = 0, m;
    while ((m = re.exec(passage)) !== null) {
      parent.appendChild(document.createTextNode(passage.slice(last, m.index)));
      parent.appendChild(el("mark", "", m[1]));
      last = m.index + m[0].length;
    }
    parent.appendChild(document.createTextNode(passage.slice(last)));
  }

  function layout() {
    const pad = 24, rx = (maxX - minX) || 1, ry = (maxY - minY) || 1;
    const s = Math.min((W - 2 * pad) / rx, (H - 2 * pad) / ry);
    const ox = (W - rx * s) / 2, oy = (H - ry * s) / 2;
    pts.forEach(p => { p.sx = ox + (p.x - minX) * s; p.sy = H - (oy + (p.y - minY) * s); });
  }

  function shape(kind, x, y, r) {
    ctx.beginPath();
    if (kind === "circle") { ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); }
    else if (kind === "square") { ctx.fillRect(x - r, y - r, 2 * r, 2 * r); }
    else if (kind === "triangle") {
      ctx.moveTo(x, y - r * 1.2); ctx.lineTo(x + r * 1.1, y + r * .9); ctx.lineTo(x - r * 1.1, y + r * .9);
      ctx.closePath(); ctx.fill();
    } else if (kind === "diamond") {
      ctx.moveTo(x, y - r * 1.3); ctx.lineTo(x + r * 1.1, y); ctx.lineTo(x, y + r * 1.3); ctx.lineTo(x - r * 1.1, y);
      ctx.closePath(); ctx.fill();
    } else {
      ctx.lineWidth = Math.max(1.5, r * .6);
      if (kind === "plus") { ctx.moveTo(x - r, y); ctx.lineTo(x + r, y); ctx.moveTo(x, y - r); ctx.lineTo(x, y + r); }
      else { ctx.moveTo(x - r, y - r); ctx.lineTo(x + r, y + r); ctx.moveTo(x + r, y - r); ctx.lineTo(x - r, y + r); }
      ctx.stroke();
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    const r = +sizeEl.value;
    for (const pass of [0, 1]) {            // noise first, clusters on top
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        if ((p.c < 0 ? 0 : 1) !== pass || !visible(p)) continue;
        const dim = focus !== null && p.c !== focus;
        ctx.globalAlpha = dim ? 0.08 : (p.c < 0 ? 0.45 : 0.85);
        ctx.fillStyle = ctx.strokeStyle = colorOf(p);
        shape(shapeOf(p), p.sx, p.sy, r);
      }
    }
    ctx.globalAlpha = 1;
    if (hover >= 0) {
      const p = pts[hover];
      ctx.lineWidth = 2;
      ctx.strokeStyle = getComputedStyle(document.body).color;
      ctx.beginPath(); ctx.arc(p.sx, p.sy, r + 4, 0, 6.2832); ctx.stroke();
    }
  }

  function resize() {
    const b = cv.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = b.width; H = b.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout(); draw();
  }
  new ResizeObserver(resize).observe(cv);

  // ---- hover tooltip ----
  cv.addEventListener("mousemove", e => {
    const b = cv.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top;
    let best = -1, bd = 169;
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (!visible(p) || (focus !== null && p.c !== focus)) continue;
      const d = (p.sx - mx) ** 2 + (p.sy - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best !== hover) { hover = best; draw(); }
    if (best < 0) { tip.style.display = "none"; return; }
    const p = pts[best];
    tip.replaceChildren();
    const meta = el("div", "meta",
      `${terms[p.t]} · ${p.c < 0 ? "noise" : "cluster " + p.c}` +
      (p.yr ? ` · ${p.yr}` : "") + ` · p=${p.pr.toFixed(2)}`);
    tip.appendChild(meta);
    if (p.ti) tip.appendChild(el("div", "meta", `${p.ti.slice(0, 90)} (${p.d})`));
    if (p.co) tip.appendChild(el("div", "meta", "collocates: " + p.co.split(";").join(", ")));
    const body = el("div"); renderPassage(body, p.p); tip.appendChild(body);
    tip.style.display = "block";
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - w - 8) + "px";
    tip.style.top = Math.min(e.clientY + 14, window.innerHeight - h - 8) + "px";
  });
  cv.addEventListener("mouseleave", () => { hover = -1; tip.style.display = "none"; draw(); });

  // ---- legends ----
  function chip(label, swatchColor, glyph, key, set, onToggle) {
    const c = el("span", "chip" + (set.has(key) ? " off" : ""));
    if (glyph) c.appendChild(el("span", "", glyph)); else {
      const s = el("span", "sw"); s.style.background = swatchColor; c.appendChild(s);
    }
    c.appendChild(document.createTextNode(label));
    c.addEventListener("click", () => {
      set.has(key) ? set.delete(key) : set.add(key);
      c.classList.toggle("off"); onToggle();
    });
    return c;
  }

  function buildLegends() {
    const cl = document.getElementById("colorLegend"); cl.replaceChildren();
    if (mode === "cluster") {
      const ids = [...new Set(pts.map(p => p.c))].sort((a, b) => a - b);
      ids.forEach(c => {
        const n = pts.filter(p => p.c === c).length;
        cl.appendChild(chip(`${c < 0 ? "noise" : "cluster " + c} (${n})`,
          c < 0 ? NOISE : PAL[c % PAL.length], null, "c" + c, hiddenColor, draw));
      });
    } else {
      terms.forEach((t, i) => {
        const n = pts.filter(p => p.t === i).length;
        cl.appendChild(chip(`${t} (${n})`, PAL[i % PAL.length], null, "t" + i, hiddenColor, draw));
      });
    }
    const sl = document.getElementById("shapeLegend"); sl.replaceChildren();
    terms.forEach((t, i) => sl.appendChild(
      chip(t, null, GLYPH[SHAPES[i % SHAPES.length]], i, hiddenTerm, draw)));
  }

  // ---- cluster cards ----
  function buildCards() {
    const box = document.getElementById("cards"); box.replaceChildren();
    summary.forEach(s => {
      const card = el("div", "card" + (focus === s.cluster ? " sel" : ""));
      const h = el("h3");
      const sw = el("span", "sw"); sw.style.background = s.cluster < 0 ? NOISE : PAL[s.cluster % PAL.length];
      h.appendChild(sw);
      h.appendChild(document.createTextNode(s.cluster < 0 ? "Noise" : "Cluster " + s.cluster));
      card.appendChild(h);
      card.appendChild(el("div", "sub",
        `${s.n} windows · ${s.docs} docs` + (s.yr ? ` · median ${s.yr}` : "")));
      if (s.dc) {
        const tags = el("div", "tags");
        s.dc.split(";").filter(Boolean).forEach(x => {
          const k = x.lastIndexOf(":");
          tags.appendChild(el("span", "tag", k > 0 ? `${x.slice(0, k)} (${x.slice(k + 1)})` : x));
        });
        card.appendChild(tags);
      }
      const parts = (s.tc || "").split(";").filter(Boolean).map(x => {
        const k = x.lastIndexOf(":"); return { t: x.slice(0, k), n: +x.slice(k + 1) };
      });
      const total = parts.reduce((a, b) => a + b.n, 0) || 1;
      const bar = el("div", "bar");
      parts.forEach(pt => {
        const seg = el("div"); const i = terms.indexOf(pt.t);
        seg.style.width = (100 * pt.n / total) + "%";
        seg.style.background = PAL[(i < 0 ? 0 : i) % PAL.length];
        seg.title = `${pt.t}: ${pt.n}`; bar.appendChild(seg);
      });
      card.appendChild(bar);
      card.appendChild(el("div", "bartxt", parts.map(pt => `${pt.t} ${pt.n}`).join(" · ")));
      card.addEventListener("click", () => { focus = focus === s.cluster ? null : s.cluster; buildCards(); draw(); });
      box.appendChild(card);
    });
  }

  document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener("change", e => {
    mode = e.target.value; hiddenColor.clear(); buildLegends(); draw();
  }));
  sizeEl.addEventListener("input", draw);
  document.getElementById("reset").addEventListener("click", () => {
    hiddenColor.clear(); hiddenTerm.clear(); focus = null; buildLegends(); buildCards(); draw();
  });

  const nClusters = summary.filter(s => s.cluster >= 0).length;
  document.getElementById("counts").textContent =
    `${pts.length} windows · ${terms.length} terms · ${nClusters} clusters · ` +
    `${pts.filter(p => p.c < 0).length} noise. Colour = cluster/term, shape = term. Hover for the passage.`;
  buildLegends(); buildCards(); resize();
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
