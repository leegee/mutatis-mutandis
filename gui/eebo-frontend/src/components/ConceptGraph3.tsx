/**
 * ConceptGraph3.tsx
 *
 * Visualises event neighbourhood topology from FAISS-derived KNN data.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * CORE IDEA
 * ─────────────────────────────────────────────────────────────────────────────
 * Each node  = one corpus event occurrence (a single use of the concept word).
 * Each edge  = cosine similarity between two events' distributional signatures.
 *
 * A distributional signature is a normalised histogram of the KNN neighbour
 * *scores* (cosine similarities) for that event, rescaled to the observed
 * [min, max] range across the current filtered event set before binning.
 * Rescaling is essential: raw FAISS cosine scores cluster tightly around
 * 0.90 (std ≈ 0.06), so binning over [0, 1] collapses everything into one
 * or two bins, making all signatures identical.  Rescaling spreads the actual
 * variance across the full histogram so signatures genuinely differ.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY NOT RAW KNN OVERLAP (Direction 1)?
 * ─────────────────────────────────────────────────────────────────────────────
 * Raw overlap counts shared neighbour event_ids.  In practice FAISS neighbour
 * lists are dominated by the same dense attractor clusters, so overlap is
 * near-uniform across all pairs.  Edge weights have almost no variance and the
 * graph is either fully dense or empty depending on the threshold — neither
 * is useful.
 *
 * Distributional signatures sidestep neighbour identity entirely.  Two events
 * can have identical histograms even if they share no neighbour IDs at all.
 * The resulting graphs are genuinely sparse and structured.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PIPELINE
 * ─────────────────────────────────────────────────────────────────────────────
 *   Tier2Data (props.data)
 *       │  filterByYearRange()        — optional, driven by year slider
 *       ▼
 *   ConceptEvent[]                    — filtered event list
 *       │  buildDistributionalGraph() — O(E² × B), reruns on filter change
 *       ▼
 *   DistGraphData { DistNode[], DistEdge[] }
 *       │  render()
 *       ▼
 *   SVG (D3 force simulation)
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * CONTROLS
 * ─────────────────────────────────────────────────────────────────────────────
 *   Concept         — which concept word to visualise
 *   Max nodes       — top-N events by degree after thresholding
 *   Min similarity  — cosine similarity threshold between signatures [0,1]
 *   Bins            — number of histogram bins (resolution of signature)
 *   Year mode       — single year or range
 *   Year slider(s)  — filter events to a publication-year window
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * DRILL-DOWN PANEL
 * ─────────────────────────────────────────────────────────────────────────────
 * Clicking a node opens a panel showing:
 *   - token, year, doc_id (with link to /api/doc/<id>)
 *   - inline bar chart of the node's score histogram
 *   - raw FAISS neighbour list (token + cosine score) sorted descending
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * INVARIANTS
 * ─────────────────────────────────────────────────────────────────────────────
 *   - Node ids are stable string keys (stringified event_id or "idx:N")
 *   - All edges are fully materialised (source/target are DistNode objects)
 *     before D3 sees them — no runtime string→node coercion inside the sim
 *   - D3 simulation holds no references to SolidJS reactivity
 */

import {
  createSignal,
  createMemo,
  createEffect,
  onCleanup,
  For,
  Show,
  type Component,
} from "solid-js";

import * as d3 from "d3";

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

interface Neighbour {
  token: string;
  score: number;
  event_id?: number;
  doc_id?: string;
  pub_year?: number;
  window_id?: number;
}

interface ConceptEvent {
  event_id?: number;
  token?: string;
  doc_id?: string;
  pub_year?: number;
  neighbours: Neighbour[];
}

interface ConceptData {
  n_events: number;
  year_min?: number;
  year_max?: number;
  events: ConceptEvent[];
}

export interface Tier2Data {
  [concept: string]: ConceptData;
}

/** A single event node in the distributional graph. */
interface DistNode extends d3.SimulationNodeDatum {
  /** Stable unique key: stringified event_id, or "idx:N" if absent. */
  id: string;
  /** The concept token this event is an occurrence of. */
  token: string;
  /** Publication year, carried for label rendering and drill-down. */
  pub_year?: number;
  /** Source document id, for drill-down linking. */
  doc_id?: string;
  /**
   * L2-normalised score histogram over nBins equal-width buckets in [0, 1].
   * This is the distributional signature used for cosine comparison.
   */
  signature: Float32Array;
  /** Post-threshold degree (number of retained edges). */
  degree: number;
  /** Raw FAISS neighbours, carried for the drill-down panel. */
  neighbours: Neighbour[];
}

interface DistEdge extends d3.SimulationLinkDatum<DistNode> {
  source: DistNode;
  target: DistNode;
  /** Cosine similarity between source.signature and target.signature. */
  weight: number;
}

interface DistGraphData {
  nodes: DistNode[];
  edges: DistEdge[];
  maxWeight: number;
  maxDegree: number;
}

interface Props {
  data: Tier2Data;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const CORPUS_START_YEAR = 1625;
const CORPUS_END_YEAR = 1665;

const EMPTY_GRAPH: DistGraphData = {
  nodes: [],
  edges: [],
  maxWeight: 1,
  maxDegree: 1,
};

// ─────────────────────────────────────────────────────────────────────────────
// Data functions
// ─────────────────────────────────────────────────────────────────────────────

function scanYearRange(cd: ConceptData): [number | undefined, number | undefined] {
  let min: number | undefined;
  let max: number | undefined;
  for (const e of cd.events) {
    const y = e.pub_year;
    if (y === undefined) continue;
    if (min === undefined || y < min) min = y;
    if (max === undefined || y > max) max = y;
  }
  return [min, max];
}

function filterByYearRange(
  events: ConceptEvent[],
  fromYear: number,
  toYear: number
): ConceptEvent[] {
  return events.filter(
    (e) => e.pub_year !== undefined && e.pub_year >= fromYear && e.pub_year <= toYear
  );
}

/**
 * Build an L2-normalised score histogram for one event's neighbour list.
 *
 * Scores are first rescaled from [scoreMin, scoreMax] → [0, 1] using the
 * corpus-wide observed range passed in by the caller.  This is critical:
 * FAISS cosine scores cluster tightly (mean ≈ 0.90, std ≈ 0.06), so binning
 * over the nominal [0, 1] range collapses every score into the same 1–2 bins,
 * making all signatures identical and all pairwise similarities → 1.0.
 * Rescaling spreads the actual variance across the full bin range.
 *
 * An all-zero vector is returned when the neighbour list is empty; such nodes
 * are excluded from the graph before edge computation.
 */
function buildSignature(
  scores: number[],
  nBins: number,
  scoreMin: number,
  scoreMax: number
): Float32Array {
  const hist = new Float32Array(nBins);
  if (scores.length === 0) return hist;

  const range = scoreMax - scoreMin;

  for (const s of scores) {
    // Rescale to [0, 1] within the observed range; guard against zero range
    const rescaled = range > 1e-9 ? (s - scoreMin) / range : 0.5;
    const clamped = Math.min(1, Math.max(0, rescaled));
    const bin = Math.min(nBins - 1, Math.floor(clamped * nBins));
    hist[bin] += 1;
  }

  let norm = 0;
  for (let i = 0; i < nBins; i++) norm += hist[i] * hist[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < nBins; i++) hist[i] /= norm;

  return hist;
}

/** Cosine similarity between two L2-normalised histograms of equal length. */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return Math.min(1, Math.max(0, dot));
}

/**
 * Build a D3-ready distributional graph from a filtered event list.
 *
 * 1. Assign a stable string key to each event.
 * 2. Build a normalised score histogram (signature) per event.
 *    Events with no neighbours are skipped — they can never form edges.
 * 3. Compute pairwise cosine similarity (upper triangle, O(E² × B)).
 * 4. Emit edges where similarity ≥ minSimilarity.
 * 5. Degree-sort all connected nodes, retain top maxNodes.
 * 6. Materialise DistNode and DistEdge objects with pre-resolved references
 *    so D3 never receives raw string source/target values.
 */
function buildDistributionalGraph(
  events: ConceptEvent[],
  minSimilarity: number,
  maxNodes: number,
  nBins: number
): DistGraphData {
  if (events.length === 0) return EMPTY_GRAPH;

  // Step 1 — scan corpus-wide score range across all events in this filtered set.
  // Must happen before signature building so every event is rescaled identically.
  let scoreMin = Infinity;
  let scoreMax = -Infinity;
  for (const event of events) {
    for (const nb of event.neighbours) {
      if (nb.score < scoreMin) scoreMin = nb.score;
      if (nb.score > scoreMax) scoreMax = nb.score;
    }
  }
  // Fallback if no neighbours at all
  if (!isFinite(scoreMin)) { scoreMin = 0; scoreMax = 1; }

  console.log("[dist-graph] score range =", scoreMin.toFixed(4), "–", scoreMax.toFixed(4));

  // Step 2 — keys and signatures (scores rescaled to observed range)
  type Meta = { key: string; event: ConceptEvent; signature: Float32Array };
  const metas: Meta[] = [];

  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const key = event.event_id !== undefined ? String(event.event_id) : `idx:${ idx }`;
    const scores = event.neighbours.map((n) => n.score);
    if (scores.length === 0) continue;
    metas.push({ key, event, signature: buildSignature(scores, nBins, scoreMin, scoreMax) });
  }

  console.log("[dist-graph] events with signatures =", metas.length,
    "| nBins =", nBins, "| minSimilarity =", minSimilarity);

  // Step 3 & 4 — pairwise cosine, upper triangle
  const rawEdges: Array<[string, string, number]> = [];

  for (let i = 0; i < metas.length; i++) {
    for (let j = i + 1; j < metas.length; j++) {
      const sim = cosineSimilarity(metas[i].signature, metas[j].signature);
      if (sim >= minSimilarity) rawEdges.push([metas[i].key, metas[j].key, sim]);
    }
  }

  console.log("[dist-graph] edges above threshold =", rawEdges.length);

  // Step 5 — degree accumulation & node selection
  const degreeMap = new Map<string, number>();
  for (const [a, b] of rawEdges) {
    degreeMap.set(a, (degreeMap.get(a) ?? 0) + 1);
    degreeMap.set(b, (degreeMap.get(b) ?? 0) + 1);
  }

  if (degreeMap.size === 0) {
    console.log("[dist-graph] no connected nodes — EMPTY_GRAPH");
    return EMPTY_GRAPH;
  }

  const topEntries = [...degreeMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxNodes);

  const keepSet = new Set(topEntries.map(([id]) => id));
  const metaByKey = new Map(metas.map((m) => [m.key, m]));

  // Step 6 — materialise nodes (DistNode objects with all provenance)
  const nodes: DistNode[] = topEntries.map(([id, degree]) => {
    const { event, signature } = metaByKey.get(id)!;
    return {
      id,
      token: event.token ?? id,
      pub_year: event.pub_year,
      doc_id: event.doc_id,
      signature,
      degree,
      neighbours: event.neighbours,
    };
  });

  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  // Materialise edges — only between retained nodes, source/target are objects
  const edges: DistEdge[] = rawEdges
    .filter(([a, b]) => keepSet.has(a) && keepSet.has(b))
    .map(([a, b, weight]) => ({
      source: nodeIndex.get(a)!,
      target: nodeIndex.get(b)!,
      weight,
    }));

  console.log("[dist-graph] final nodes =", nodes.length, "| final edges =", edges.length);

  return {
    nodes,
    edges,
    maxWeight: Math.max(1, ...edges.map((e) => e.weight)),
    maxDegree: Math.max(1, ...nodes.map((n) => n.degree)),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

const ConceptGraph3: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxNodes, setMaxNodes] = createSignal(50);
  const [minSimilarity, setMinSimilarity] = createSignal(0.7);
  const [nBins, setNBins] = createSignal(10);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);
  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);
  const [yearMode, setYearMode] = createSignal<"single" | "range">("range");

  // Year bounds derived from the current concept's events
  const yearBounds = createMemo<[number, number]>(() => {
    const cd = props.data[concept()];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    const [min, max] = scanYearRange(cd);
    return [min ?? CORPUS_START_YEAR, max ?? CORPUS_END_YEAR];
  });

  // Reset year sliders when concept or year mode changes
  createEffect(() => {
    const [min, max] = yearBounds();
    if (yearMode() === "single") {
      const mid = Math.floor((min + max) / 2);
      setFromYear(mid);
      setToYear(mid);
    } else {
      setFromYear(min);
      setToYear(max);
    }
  });

  // Temporally filtered event list
  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    const events = cd.events;
    const filtered =
      fromYear() <= min && toYear() >= max
        ? events
        : filterByYearRange(events, fromYear(), toYear());
    console.log("[year-filter] range:", fromYear(), "–", toYear(),
      "| in:", events.length, "→ out:", filtered.length);
    return filtered;
  });

  // Graph — reruns whenever filtered events, threshold, maxNodes, or nBins change
  const graphData = createMemo<DistGraphData>(() =>
    buildDistributionalGraph(yearFiltered(), minSimilarity(), maxNodes(), nBins())
  );

  // Selected node object (resolved from graph, not a secondary map lookup)
  const selectedDistNode = createMemo<DistNode | null>(() => {
    const id = selectedNode();
    if (!id) return null;
    return graphData().nodes.find((n) => n.id === id) ?? null;
  });

  // Neighbours sorted by score descending for the drill-down panel
  const selectedNeighbours = createMemo(() =>
    [...(selectedDistNode()?.neighbours ?? [])].sort((a, b) => b.score - a.score)
  );

  // ── D3 ──────────────────────────────────────────────────────────────────────

  let svgRef!: SVGSVGElement;
  let simulationRef: d3.Simulation<DistNode, DistEdge> | null = null;

  function render() {
    const { nodes, edges, maxWeight, maxDegree } = graphData();
    const svg = d3.select(svgRef);
    const W = svgRef.clientWidth;
    const H = svgRef.clientHeight;

    svg.selectAll("*").remove();

    if (nodes.length === 0) {
      svg
        .append("text")
        .attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "rgb(205, 89, 89)")
        .attr("font-size", "2rem")
        .attr("font-family", "monospace")
        .text("No graph — try reducing min similarity");
      return;
    }

    const nodeRadius = d3.scaleSqrt().domain([0, maxDegree]).range([4, 18]);
    const edgeOpacity = d3.scaleLinear().domain([0, maxWeight]).range([0.5, 1]);
    const edgeWidth = d3.scaleLinear().domain([0, maxWeight]).range([2, 10]);
    const nodeColor = d3.scaleLinear<string>()
      .domain([0, maxDegree])
      .range(["#5a87ba66", "#e9f3fcdd"]);

    const container = svg.append("g").attr("class", "zoom-container");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => container.attr("transform", event.transform))
    );

    const defs = container.append("defs");

    edges.forEach((d, i) => {
      const grad = defs.append("linearGradient")
        .attr("id", `eg-${ i }`)
        .attr("gradientUnits", "userSpaceOnUse");
      grad.append("stop").attr("offset", "0%").attr("stop-color", nodeColor(d.source.degree));
      grad.append("stop").attr("offset", "100%").attr("stop-color", nodeColor(d.target.degree));
    });

    const edgeSelection = container
      .append("g").attr("class", "edges")
      .selectAll<SVGLineElement, DistEdge>("line")
      .data(edges).join("line")
      .attr("stroke", (_, i) => `url(#eg-${ i })`)
      .attr("stroke-opacity", (d) => edgeOpacity(d.weight))
      .attr("stroke-width", (d) => edgeWidth(d.weight));

    const nodeGroup = container
      .append("g").attr("class", "nodes")
      .selectAll<SVGGElement, DistNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode((prev) => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGGElement, DistNode>()
          .on("start", (event, d) => {
            if (!event.active) simulationRef?.alphaTarget(0.3).restart();
            d.fx = d.x; d.fy = d.y;
          })
          .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
          .on("end", (event, d) => {
            if (!event.active) simulationRef?.alphaTarget(0);
            d.fx = null; d.fy = null;
          })
      );

    nodeGroup.append("circle")
      .attr("r", (d) => nodeRadius(d.degree))
      .attr("fill", (d) => nodeColor(d.degree))
      .attr("stroke", "rgba(200,230,255,0.25)")
      .attr("stroke-width", 1);

    // Label: short doc_id fragment + year — token is uniform per concept so useless as a label
    nodeGroup.append("text")
      .text((d) => {
        const doc = d.doc_id ? d.doc_id.slice(-8) : d.id;
        return d.pub_year !== undefined ? `${ doc } · ${ d.pub_year }` : doc;
      })
      .attr("dy", (d) => -nodeRadius(d.degree) - 3)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("font-family", "'IBM Plex Mono', 'Courier New', monospace")
      .attr("fill", "rgba(210,235,255,0.85)")
      .attr("pointer-events", "none");

    const tooltip = d3.select("body")
      .selectAll<HTMLDivElement, unknown>(".cg-tooltip")
      .data([null]).join("div")
      .attr("class", "cg-tooltip surface-container-high border large-elevate padding")
      .style("position", "fixed")
      .style("pointer-events", "none")
      .style("font-family", "'IBM Plex Mono', monospace")
      .style("opacity", "0")
      .style("transition", "opacity 0.15s");

    nodeGroup
      .on("mouseenter", (event, d) => {
        tooltip
          .html(
            `<aside>` +
            `<h6 class="bottom-padding">${ d.doc_id ?? d.id }${ d.pub_year !== undefined ? ` · ${ d.pub_year }` : "" }</h6>` +
            `Token: ${ d.token }<br/>` +
            `Connections: ${ d.degree }<br/>` +
            `Neighbours: ${ d.neighbours.length }` +
            `</aside>`
          )
          .style("opacity", "1")
          .style("left", event.clientX + 14 + "px")
          .style("top", event.clientY - 10 + "px");
      })
      .on("mousemove", (event) => {
        tooltip.style("left", event.clientX + 14 + "px").style("top", event.clientY - 10 + "px");
      })
      .on("mouseleave", () => tooltip.style("opacity", "0"));

    if (simulationRef) simulationRef.stop();

    simulationRef = d3.forceSimulation<DistNode>(nodes)
      .force("link",
        d3.forceLink<DistNode, DistEdge>(edges)
          .id((d) => d.id)
          // Higher cosine similarity → pull nodes closer together
          .distance((d) => Math.max(50, 220 - d.weight * 160))
          .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide<DistNode>().radius((d) => nodeRadius(d.degree) + 6))
      .on("tick", () => {
        edgeSelection
          .attr("x1", (d) => (d.source as DistNode).x ?? 0)
          .attr("y1", (d) => (d.source as DistNode).y ?? 0)
          .attr("x2", (d) => (d.target as DistNode).x ?? 0)
          .attr("y2", (d) => (d.target as DistNode).y ?? 0);

        edges.forEach((d, i) => {
          defs.select(`#eg-${ i }`)
            .attr("x1", d.source.x ?? 0).attr("y1", d.source.y ?? 0)
            .attr("x2", d.target.x ?? 0).attr("y2", d.target.y ?? 0);
        });

        nodeGroup.attr("transform", (d) => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`);
      });
  }

  createEffect(() => { graphData(); if (svgRef) render(); });

  onCleanup(() => {
    simulationRef?.stop();
    d3.select("body").selectAll(".cg-tooltip").remove();
  });

  // ── UI ──────────────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%", width: "100%" }}>

      <header class="center-align fill max surface-container-low small-padding top-padding">
        <nav>

          {/* Concept */}
          <div class="field suffix border middle-align">
            <select value={concept()} onChange={(e) => setConcept(e.currentTarget.value)}>
              <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
            </select>
            <output>Concept</output>
          </div>

          {/* Max nodes */}
          <div class="field suffix border middle-align">
            <select value={maxNodes()} onChange={(e) => setMaxNodes(Number(e.currentTarget.value))}>
              <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
            </select>
            <output>Max nodes</output>
          </div>

          {/* Min similarity */}
          <div class="field middle-align">
            <div class="slider tiny">
              <input
                type="range" min={0.05} max={0.95} step={0.05}
                value={minSimilarity()}
                onInput={(e) => setMinSimilarity(Number(e.currentTarget.value))}
              />
              <span />
              <span class="tooltip bottom" />
            </div>
            <output class="small-padding top-padding">
              Min similarity {minSimilarity().toFixed(2)}
            </output>
          </div>

          {/* Bins */}
          <div class="field suffix border middle-align">
            <select value={nBins()} onChange={(e) => setNBins(Number(e.currentTarget.value))}>
              <For each={[5, 10, 20]}>{(n) => <option value={n}>{n}</option>}</For>
            </select>
            <output>Bins</output>
          </div>

          {/* Year mode */}
          <div class="field suffix border middle-align">
            <select
              value={yearMode()}
              onChange={(e) => setYearMode(e.currentTarget.value as "single" | "range")}
            >
              <option value="single">Single year</option>
              <option value="range">Year range</option>
            </select>
            <output>Year mode</output>
          </div>

          {/* Single-year slider */}
          <Show when={yearMode() === "single"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range" min={CORPUS_START_YEAR} max={CORPUS_END_YEAR} step={1}
                  value={fromYear()}
                  onInput={(e) => {
                    const v = Number(e.currentTarget.value);
                    setFromYear(v); setToYear(v);
                  }}
                />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                {fromYear()} ({yearFiltered().length} events)
              </output>
            </div>
          </Show>

          {/* Year-range sliders */}
          <Show when={yearMode() === "range"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                  value={fromYear()}
                  onInput={(e) => setFromYear(Math.min(Number(e.currentTarget.value), toYear()))}
                />
                <input
                  type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                  value={toYear()}
                  onInput={(e) => setToYear(Math.max(Number(e.currentTarget.value), fromYear()))}
                />
                <span />
                <span class="tooltip bottom" />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                <span>{fromYear()}–{toYear()}</span>
                <span class="left-padding">
                  {yearFiltered().length}/{props.data[concept()]?.n_events ?? 0} events
                </span>
              </output>
            </div>
          </Show>

        </nav>
      </header>

      {/* Main area */}
      <div style={{ display: "flex", flex: "1", overflow: "hidden" }} class="background">

        <svg ref={svgRef!} style={{ flex: "1", display: "block" }} class="surface-container-lowest" />

        {/* Drill-down panel */}
        <Show when={selectedNode()}>
          <aside
            class="surface-container-high padding border"
            style={{ width: "20rem", "flex-shrink": "0", "overflow-y": "auto" }}
          >
            <div style={{ display: "flex", "justify-content": "space-between", "align-items": "center" }}>
              <h2>
                {selectedDistNode()?.doc_id ?? selectedDistNode()?.id ?? selectedNode()}
                <Show when={selectedDistNode()?.pub_year !== undefined}>
                  <span class="small-text"> · {selectedDistNode()!.pub_year}</span>
                </Show>
              </h2>
              <button class="link border" onClick={() => setSelectedNode(null)}>✕</button>
            </div>

            <Show when={selectedDistNode()}>
              {(node) => (
                <div class="bottom-padding">
                  <div>Event ID: {node().id}</div>
                  <div>Connections: {node().degree}</div>
                  <Show when={node().doc_id}>
                    <button class="chip small-margin" onClick={() => showDocument(node().doc_id!)}>
                      <span>{node().doc_id}</span>
                      <Show when={node().pub_year !== undefined}>
                        <span class="small-text"> {node().pub_year}</span>
                      </Show>
                    </button>
                  </Show>
                </div>
              )}
            </Show>

            {/* Score histogram — inline bar chart of the distributional signature */}
            <h3 class="bottom-padding">Score distribution</h3>
            <Show when={selectedDistNode()}>
              {(node) => {
                const sig = node().signature;
                const maxBin = Math.max(...sig);
                const binWidth = 1 / sig.length;
                return (
                  <div style={{ display: "flex", "align-items": "flex-end", gap: "2px", height: "48px", "margin-bottom": "0.75rem" }}>
                    <For each={Array.from(sig)}>
                      {(v, i) => (
                        <div
                          title={`bin ${ i() + 1 }/${ sig.length } (rescaled): ${ v.toFixed(3) }`}
                          style={{
                            flex: "1",
                            height: maxBin > 0 ? `${ (v / maxBin) * 100 }%` : "0%",
                            background: "rgba(100,180,255,0.6)",
                            "border-radius": "2px 2px 0 0",
                            "min-height": v > 0 ? "2px" : "0",
                          }}
                        />
                      )}
                    </For>
                  </div>
                );
              }}
            </Show>

            {/* Raw FAISS neighbour list */}
            <h3 class="bottom-padding">FAISS neighbours</h3>
            <Show
              when={selectedNeighbours().length > 0}
              fallback={<div class="error">No neighbours</div>}
            >
              <For each={selectedNeighbours()}>
                {(nb) => (
                  <div class="small-margin" style={{ display: "flex", "justify-content": "space-between" }}>
                    <span style={{ "font-family": "'IBM Plex Mono', monospace", "font-size": "0.85rem" }}>
                      {nb.token}
                    </span>
                    <span class="small-text" style={{ color: "rgba(180,210,255,0.7)" }}>
                      {nb.score.toFixed(4)}
                      <Show when={nb.pub_year !== undefined}>{" "}· {nb.pub_year}</Show>
                    </span>
                  </div>
                )}
              </For>
            </Show>

          </aside>
        </Show>

      </div>

      <footer
        class="fixed max center-align small-padding surface-container-low"
        style={{ "flex-shrink": "0" }}
      >
        {graphData().nodes.length} nodes
        {" • "}
        {graphData().edges.length} edges
        {" • "}
        {yearFiltered().length} events
        <Show when={fromYear() !== yearBounds()[0] || toYear() !== yearBounds()[1]}>
          {" • "}
          {fromYear()}–{toYear()}
        </Show>
      </footer>

    </div>
  );
};

export default ConceptGraph3;