/**
 * ContextGraph.tsx
 *
 * Token-binned contextual similarity graph with neighbour expansion.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * NODE KINDS
 * ─────────────────────────────────────────────────────────────────────────────
 *  "hub"       — one per distinct surface form (LAW, LAWES …) aggregated
 *                across all events in the current year window.
 *                Radius ∝ sqrt(eventCount).  Filled circle.
 *
 *  "neighbour" — one per distinct neighbour token appearing in any hub's
 *                top-N list.  Shared across hubs: if PARLIAMENT appears in
 *                both LAW's and PREROGATIVE's top neighbours it is one node
 *                with two spokes.  Fixed small radius.  Diamond shape.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * EDGE KINDS
 * ─────────────────────────────────────────────────────────────────────────────
 *  "hub-hub"       — cosine similarity between two hubs' normalised
 *                    neighbour-frequency vectors.  Solid gradient line.
 *                    Only drawn when similarity ≥ minSimilarity.
 *                    Isolated hubs (no hub-hub edges) are still shown
 *                    because they carry spoke edges.
 *
 *  "hub-neighbour" — spoke from hub to each of its top-N neighbours.
 *                    Weight = normalised frequency of that neighbour in
 *                    the hub's vector.  Dashed, lower opacity.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * PIPELINE
 * ─────────────────────────────────────────────────────────────────────────────
 *   props.data (Tier2Data)
 *       │  filterByYearRange()
 *       ▼
 *   ConceptEvent[]
 *       │  aggregateByToken()   — bins events, normalised neighbour-freq vectors
 *       ▼
 *   Map<token, TokenBin>
 *       │  buildContextualGraph(topN, minSimilarity, maxNodes)
 *       │    1. hub-hub edges: pairwise cosine ≥ minSimilarity
 *       │    2. neighbour nodes: union of top-N lists, shared across hubs
 *       │    3. hub-neighbour edges: spoke per (hub, neighbour) pair
 *       │    4. all isolated hubs retained (they have spokes)
 *       ▼
 *   ContextGraphData { nodes, hubHubEdges, hubNbEdges }
 *       │  render()
 *       ▼
 *   SVG — two edge layers, two node layers
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * DRILL-DOWN
 * ─────────────────────────────────────────────────────────────────────────────
 *  Hub node    → event count, doc range, year range, source doc chips
 *  Neighbour   → "shared by" hub list (rhetorical coalition signal) +
 *                mean cosine score per hub
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * STAGE ARCHITECTURE  (ready for layers 2 + 3)
 * ─────────────────────────────────────────────────────────────────────────────
 * TODO layer 2: temporal continuity edges between era-split hub nodes
 * TODO layer 3: shared-unusual-neighbour edges between hubs
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
// Scoped styles
// ─────────────────────────────────────────────────────────────────────────────

const STYLES = `
  .cg-layout         { display:flex; flex-direction:column; height:100%; width:100%; }
  .cg-main           { display:flex; flex:1; overflow:hidden; }
  .cg-svg            { flex:1; display:block; }
  .cg-aside          { width:22rem; flex-shrink:0; overflow-y:auto; }
  .cg-header-row     { display:flex; justify-content:space-between;
                       align-items:center; padding-bottom:.25rem; }
  .cg-nb-row         { display:flex; align-items:center; gap:.5rem;
                       padding:.3rem 0;
                       border-bottom:1px solid rgba(255,255,255,.06); }
  .cg-nb-bar-wrap    { width:4rem; flex-shrink:0; height:6px;
                       background:rgba(255,255,255,.1); border-radius:3px;
                       position:relative; overflow:hidden; }
  .cg-nb-bar-fill    { position:absolute; left:0; top:0; height:100%;
                       border-radius:3px; }
  .cg-nb-bar-fill.hub      { background:rgba(100,180,255,.75); }
  .cg-nb-bar-fill.neighbour{ background:rgba(255,190,80,.75); }
  .cg-nb-token       { flex:1; font-family:'IBM Plex Mono','Courier New',monospace;
                       font-size:.82rem; }
  .cg-nb-score       { flex-shrink:0; font-size:.75rem; opacity:.55; }
  .cg-chip-mono span { font-family:'IBM Plex Mono','Courier New',monospace;
                       font-size:.78rem; }
  .cg-legend         { display:flex; gap:1rem; align-items:center;
                       padding:.25rem .75rem; font-size:.75rem; opacity:.7; }
  .cg-legend-hub     { display:inline-block; width:10px; height:10px;
                       border-radius:50%; background:rgba(100,180,255,.8);
                       flex-shrink:0; }
  .cg-legend-nb      { display:inline-block; width:9px; height:9px;
                       background:rgba(255,190,80,.85); flex-shrink:0;
                       transform:rotate(45deg); }
`;

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

interface Props {
  data: Tier2Data;
}

interface TokenBin {
  token: string;
  eventCount: number;
  neighbourFreq: Map<string, number>;
  topNeighbours: Array<{ token: string; freq: number; meanScore: number }>;
  docs: Map<string, number | undefined>;
  years: Set<number>;
}

/** Unified simulation node for both hubs and neighbours. */
interface ContextNode extends d3.SimulationNodeDatum {
  id: string;
  kind: "hub" | "neighbour";
  /** Hub only: number of corpus events aggregated here. */
  eventCount: number;
  /** Hub only: edge count from hub-hub edges. */
  hubDegree: number;
  /** Both: total edge count (hub-hub + hub-neighbour spokes). */
  degree: number;
}

/** Hub ↔ Hub edge: cosine similarity between neighbour-freq vectors. */
interface HubHubEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-hub";
  source: ContextNode;
  target: ContextNode;
  weight: number;
}

/** Hub → Neighbour spoke: normalised frequency of that token in this hub. */
interface HubNbEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-neighbour";
  source: ContextNode;  // always the hub
  target: ContextNode;  // always the neighbour node
  weight: number;       // normalised frequency
}

type AnyEdge = HubHubEdge | HubNbEdge;

interface ContextGraphData {
  nodes: ContextNode[];
  hubHubEdges: HubHubEdge[];
  hubNbEdges: HubNbEdge[];
  allEdges: AnyEdge[];
  maxHubHubWeight: number;
  maxEventCount: number;
  maxHubDegree: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const CORPUS_START_YEAR = 1625;
const CORPUS_END_YEAR = 1665;

const HUB_COLOR_LOW = "#5a87ba66";
const HUB_COLOR_HIGH = "#e9f3fcdd";
const NB_COLOR = "rgba(255,190,80,0.85)";
const NB_RADIUS = 5;
const DIAMOND_SIZE = 7;   // half-diagonal of the diamond shape

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [], hubHubEdges: [], hubNbEdges: [], allEdges: [],
  maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1,
};

// ─────────────────────────────────────────────────────────────────────────────
// Data functions
// ─────────────────────────────────────────────────────────────────────────────

function scanYearRange(cd: ConceptData): [number, number] {
  let min = CORPUS_END_YEAR;
  let max = CORPUS_START_YEAR;
  for (const e of cd.events) {
    if (e.pub_year === undefined) continue;
    if (e.pub_year < min) min = e.pub_year;
    if (e.pub_year > max) max = e.pub_year;
  }
  return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

function filterByYearRange(
  events: ConceptEvent[], from: number, to: number
): ConceptEvent[] {
  return events.filter(
    (e) => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to
  );
}

function aggregateByToken(events: ConceptEvent[]): Map<string, TokenBin> {
  const rawCounts = new Map<string, Map<string, number>>();
  const rawScores = new Map<string, Map<string, number>>();
  const meta = new Map<string, Pick<TokenBin, "eventCount" | "docs" | "years">>();

  for (const event of events) {
    const binKey = event.token ?? "__unknown__";
    if (!rawCounts.has(binKey)) {
      rawCounts.set(binKey, new Map());
      rawScores.set(binKey, new Map());
      meta.set(binKey, { eventCount: 0, docs: new Map(), years: new Set() });
    }
    const counts = rawCounts.get(binKey)!;
    const scores = rawScores.get(binKey)!;
    const m = meta.get(binKey)!;
    m.eventCount += 1;
    if (event.doc_id) m.docs.set(event.doc_id, event.pub_year);
    if (event.pub_year !== undefined) m.years.add(event.pub_year);
    for (const nb of event.neighbours) {
      counts.set(nb.token, (counts.get(nb.token) ?? 0) + 1);
      scores.set(nb.token, (scores.get(nb.token) ?? 0) + nb.score);
    }
  }

  const bins = new Map<string, TokenBin>();
  for (const [binKey, counts] of rawCounts) {
    const scores = rawScores.get(binKey)!;
    const m = meta.get(binKey)!;
    const total = [...counts.values()].reduce((a, b) => a + b, 0);
    const neighbourFreq = new Map<string, number>();
    for (const [tok, count] of counts)
      neighbourFreq.set(tok, total > 0 ? count / total : 0);

    const topNeighbours = [...neighbourFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)   // store top-30 so the drill-down has headroom
      .map(([tok, freq]) => ({
        token: tok, freq,
        meanScore: counts.get(tok)! > 0 ? (scores.get(tok) ?? 0) / counts.get(tok)! : 0,
      }));

    bins.set(binKey, {
      token: binKey, eventCount: m.eventCount,
      neighbourFreq, topNeighbours, docs: m.docs, years: m.years,
    });
  }
  return bins;
}

function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  let normA = 0; for (const v of a.values()) normA += v * v;
  let normB = 0; for (const v of b.values()) normB += v * v;
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  const [smaller, larger] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [tok, v] of smaller) {
    const u = larger.get(tok);
    if (u !== undefined) dot += v * u;
  }
  return Math.min(1, Math.max(0, dot / (normA * normB)));
}

/**
 * Build the full two-kind graph.
 *
 * @param bins          Aggregated token bins.
 * @param topN          How many top neighbours to expand per hub.
 * @param minSimilarity Cosine threshold for hub-hub edges.
 * @param maxHubs       Max number of hub nodes to retain.
 */
function buildContextualGraph(
  bins: Map<string, TokenBin>,
  topN: number,
  minSimilarity: number,
  maxHubs: number,
): ContextGraphData {
  const hubKeys = [...bins.keys()];
  if (hubKeys.length === 0) return EMPTY_GRAPH;

  // ── 1. Hub-hub edges: pairwise cosine ─────────────────────────────────────
  const rawHubHub: Array<[string, string, number]> = [];
  for (let i = 0; i < hubKeys.length; i++) {
    for (let j = i + 1; j < hubKeys.length; j++) {
      const sim = cosineSimilarity(
        bins.get(hubKeys[i])!.neighbourFreq,
        bins.get(hubKeys[j])!.neighbourFreq,
      );
      if (sim >= minSimilarity) rawHubHub.push([hubKeys[i], hubKeys[j], sim]);
    }
  }
  // TODO layer 2: inject temporal continuity edges here
  // TODO layer 3: inject shared-unusual-neighbour edges here

  // Hub degree from hub-hub edges only
  const hubHubDegree = new Map<string, number>();
  for (const [a, b] of rawHubHub) {
    hubHubDegree.set(a, (hubHubDegree.get(a) ?? 0) + 1);
    hubHubDegree.set(b, (hubHubDegree.get(b) ?? 0) + 1);
  }

  // All hubs are retained (isolated hubs still get spokes).
  // Sort by hub-hub degree desc then eventCount desc so the most connected
  // hubs are preferred when maxHubs truncates.
  const sortedHubs = hubKeys
    .sort((a, b) => {
      const dd = (hubHubDegree.get(b) ?? 0) - (hubHubDegree.get(a) ?? 0);
      if (dd !== 0) return dd;
      return bins.get(b)!.eventCount - bins.get(a)!.eventCount;
    })
    .slice(0, maxHubs);

  const hubSet = new Set(sortedHubs);

  // ── 2. Hub nodes ───────────────────────────────────────────────────────────
  const nodeMap = new Map<string, ContextNode>();

  for (const key of sortedHubs) {
    nodeMap.set(key, {
      id: key,
      kind: "hub",
      eventCount: bins.get(key)!.eventCount,
      hubDegree: hubHubDegree.get(key) ?? 0,
      degree: hubHubDegree.get(key) ?? 0,   // will add spoke count below
    });
  }

  // ── 3. Neighbour nodes + hub-neighbour edges ───────────────────────────────
  // Collect (hubKey, nbToken, freq) for all hubs in the retained set.
  // Neighbour nodes are shared: one ContextNode per distinct token.
  const spokeTriples: Array<[string, string, number]> = [];

  for (const hubKey of sortedHubs) {
    const bin = bins.get(hubKey)!;
    const top = bin.topNeighbours.slice(0, topN);
    for (const nb of top) {
      if (!nodeMap.has(nb.token)) {
        nodeMap.set(nb.token, {
          id: nb.token, kind: "neighbour",
          eventCount: 0, hubDegree: 0, degree: 0,
        });
      }
      spokeTriples.push([hubKey, nb.token, nb.freq]);
    }
  }

  // Accumulate total degree (hub-hub + spokes)
  for (const [hubKey, nbToken] of spokeTriples) {
    const h = nodeMap.get(hubKey)!;
    const n = nodeMap.get(nbToken)!;
    h.degree += 1;
    n.degree += 1;
  }

  const nodes = [...nodeMap.values()];

  // ── 4. Materialise edges with pre-resolved node references ─────────────────
  const hubHubEdges: HubHubEdge[] = rawHubHub
    .filter(([a, b]) => hubSet.has(a) && hubSet.has(b))
    .map(([a, b, weight]) => ({
      kind: "hub-hub" as const,
      source: nodeMap.get(a)!,
      target: nodeMap.get(b)!,
      weight,
    }));

  const hubNbEdges: HubNbEdge[] = spokeTriples.map(([hubKey, nbToken, weight]) => ({
    kind: "hub-neighbour" as const,
    source: nodeMap.get(hubKey)!,
    target: nodeMap.get(nbToken)!,
    weight,
  }));

  const allEdges: AnyEdge[] = [...hubNbEdges, ...hubHubEdges];

  const maxEventCount = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.eventCount));
  const maxHubDegree = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.hubDegree));
  const maxHubHubWeight = Math.max(1, ...hubHubEdges.map(e => e.weight));

  console.log("[ctx-graph] hubs:", sortedHubs.length,
    "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length,
    "| hub-hub edges:", hubHubEdges.length,
    "| spoke edges:", hubNbEdges.length);

  return { nodes, hubHubEdges, hubNbEdges, allEdges, maxHubHubWeight, maxEventCount, maxHubDegree };
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

const ContextGraph: Component<Props> = (props) => {
  console.log(props)
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxHubs, setMaxHubs] = createSignal(50);
  const [topN, setTopN] = createSignal(5);
  const [minSimilarity, setMinSimilarity] = createSignal(0.5);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);
  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);
  const [yearMode, setYearMode] = createSignal<"single" | "range">("range");

  // ── Year bounds ────────────────────────────────────────────────────────────

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = props.data[concept()];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });

  createEffect(() => {
    const [min, max] = yearBounds();
    if (yearMode() === "single") {
      const mid = Math.floor((min + max) / 2);
      setFromYear(mid); setToYear(mid);
    } else {
      setFromYear(min); setToYear(max);
    }
  });

  // ── Filtered events ────────────────────────────────────────────────────────

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    const events = cd.events;
    return fromYear() <= min && toYear() >= max
      ? events
      : filterByYearRange(events, fromYear(), toYear());
  });

  // ── Aggregation ────────────────────────────────────────────────────────────

  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(yearFiltered())
  );

  // ── Graph ──────────────────────────────────────────────────────────────────

  const graphData = createMemo<ContextGraphData>(() =>
    buildContextualGraph(tokenBins(), topN(), minSimilarity(), maxHubs())
  );

  // ── Drill-down ─────────────────────────────────────────────────────────────

  const selectedKind = createMemo<"hub" | "neighbour" | null>(() => {
    const id = selectedNode();
    if (!id) return null;
    return graphData().nodes.find(n => n.id === id)?.kind ?? null;
  });

  // Hub drill-down: the TokenBin
  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "hub") return null;
    return tokenBins().get(id) ?? null;
  });

  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const bin = selectedBin();
    if (!bin) return [];
    return [...bin.docs.entries()].sort((a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity));
  });

  // Neighbour drill-down: which hubs share this token, with freq + meanScore
  const sharedByHubs = createMemo<Array<{ hub: string; freq: number; meanScore: number }>>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "neighbour") return [];
    const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
    for (const [hubKey, bin] of tokenBins()) {
      const nb = bin.topNeighbours.find(n => n.token === id);
      if (nb) result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore });
    }
    return result.sort((a, b) => b.freq - a.freq);
  });

  // ── D3 ────────────────────────────────────────────────────────────────────

  let svgRef!: SVGSVGElement;
  let simulationRef: d3.Simulation<ContextNode, AnyEdge> | null = null;

  function render() {
    const { nodes, hubHubEdges, hubNbEdges, allEdges,
      maxHubHubWeight, maxEventCount, maxHubDegree } = graphData();
    const svg = d3.select(svgRef);
    const W = svgRef.clientWidth;
    const H = svgRef.clientHeight;

    svg.selectAll("*").remove();

    if (nodes.length === 0) {
      svg.append("text")
        .attr("x", W / 2).attr("y", H / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "rgb(205,89,89)")
        .attr("font-size", "1.5rem")
        .attr("font-family", "'IBM Plex Mono',monospace")
        .text("No graph — try reducing min similarity or increasing top N");
      return;
    }

    const hubRadius = d3.scaleSqrt().domain([0, maxEventCount]).range([8, 30]);
    const hubColor = d3.scaleLinear<string>()
      .domain([0, Math.max(1, maxHubDegree)])
      .range([HUB_COLOR_LOW, HUB_COLOR_HIGH]);
    const hhOpacity = d3.scaleLinear().domain([0, maxHubHubWeight]).range([0.25, 0.85]);
    const hhWidth = d3.scaleLinear().domain([0, maxHubHubWeight]).range([1, 7]);
    const spokeOpacity = d3.scaleLinear().domain([0, 1]).range([0.15, 0.5]);
    const spokeWidth = d3.scaleLinear().domain([0, 1]).range([0.5, 2.5]);

    const container = svg.append("g").attr("class", "zoom-container");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (ev) => container.attr("transform", ev.transform))
    );

    const defs = container.append("defs");

    // Gradient defs for hub-hub edges only
    hubHubEdges.forEach((d, i) => {
      const grad = defs.append("linearGradient")
        .attr("id", `hh-${ i }`).attr("gradientUnits", "userSpaceOnUse");
      grad.append("stop").attr("offset", "0%")
        .attr("stop-color", hubColor(d.source.hubDegree));
      grad.append("stop").attr("offset", "100%")
        .attr("stop-color", hubColor(d.target.hubDegree));
    });

    // ── Edge layer 1: hub-neighbour spokes (rendered below hubs) ─────────────
    const spokeSelection = container.append("g").attr("class", "spokes")
      .selectAll<SVGLineElement, HubNbEdge>("line")
      .data(hubNbEdges).join("line")
      .attr("stroke", NB_COLOR)
      .attr("stroke-opacity", (d) => spokeOpacity(d.weight))
      .attr("stroke-width", (d) => spokeWidth(d.weight))
      .attr("stroke-dasharray", "3 3");

    // ── Edge layer 2: hub-hub similarity edges ────────────────────────────────
    const hhSelection = container.append("g").attr("class", "hh-edges")
      .selectAll<SVGLineElement, HubHubEdge>("line")
      .data(hubHubEdges).join("line")
      .attr("stroke", (_, i) => `url(#hh-${ i })`)
      .attr("stroke-opacity", (d) => hhOpacity(d.weight))
      .attr("stroke-width", (d) => hhWidth(d.weight));

    // ── Neighbour nodes ───────────────────────────────────────────────────────
    const nbNodes = nodes.filter(n => n.kind === "neighbour");
    const nbGroup = container.append("g").attr("class", "nb-nodes")
      .selectAll<SVGGElement, ContextNode>("g")
      .data(nbNodes, d => d.id).join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode(prev => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGGElement, ContextNode>()
          .on("start", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Diamond: rotated square
    nbGroup.append("rect")
      .attr("x", -DIAMOND_SIZE).attr("y", -DIAMOND_SIZE)
      .attr("width", DIAMOND_SIZE * 2).attr("height", DIAMOND_SIZE * 2)
      .attr("transform", "rotate(45)")
      .attr("fill", NB_COLOR)
      .attr("stroke", "rgba(255,220,120,0.3)")
      .attr("stroke-width", 1);

    nbGroup.append("text")
      .text(d => d.id)
      .attr("dx", DIAMOND_SIZE + 4)
      .attr("dy", "0.35em")
      .attr("text-anchor", "start")
      .attr("font-size", "9px")
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", "rgba(255,220,140,0.85)")
      .attr("pointer-events", "none");

    // ── Hub nodes (on top) ────────────────────────────────────────────────────
    const hubNodes = nodes.filter(n => n.kind === "hub");
    const hubGroup = container.append("g").attr("class", "hub-nodes")
      .selectAll<SVGGElement, ContextNode>("g")
      .data(hubNodes, d => d.id).join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode(prev => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGGElement, ContextNode>()
          .on("start", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    hubGroup.append("circle")
      .attr("r", d => hubRadius(d.eventCount))
      .attr("fill", d => hubColor(d.hubDegree))
      .attr("stroke", "rgba(200,230,255,0.3)")
      .attr("stroke-width", 1.5);

    // Event-count badge inside (when large enough)
    hubGroup.append("text")
      .text(d => hubRadius(d.eventCount) > 12 ? String(d.eventCount) : "")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", "9px")
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", "rgba(255,255,255,0.5)")
      .attr("pointer-events", "none");

    // Label below circle
    hubGroup.append("text")
      .text(d => d.id)
      .attr("dy", d => hubRadius(d.eventCount) + 13)
      .attr("text-anchor", "middle")
      .attr("font-size", d => Math.max(9, Math.min(13, hubRadius(d.eventCount) * 0.7)) + "px")
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", "rgba(210,235,255,0.9)")
      .attr("pointer-events", "none");

    // ── Tooltip ───────────────────────────────────────────────────────────────
    const tooltip = d3.select("body")
      .selectAll<HTMLDivElement, unknown>(".cg-tooltip")
      .data([null]).join("div")
      .attr("class", "cg-tooltip surface-container-high border large-elevate padding")
      .style("position", "fixed").style("pointer-events", "none")
      .style("font-family", "'IBM Plex Mono',monospace")
      .style("opacity", "0").style("transition", "opacity 0.15s");

    const showTip = (ev: MouseEvent, html: string) =>
      tooltip.html(html).style("opacity", "1")
        .style("left", ev.clientX + 14 + "px").style("top", ev.clientY - 10 + "px");
    const moveTip = (ev: MouseEvent) =>
      tooltip.style("left", ev.clientX + 14 + "px").style("top", ev.clientY - 10 + "px");
    const hideTip = () => tooltip.style("opacity", "0");

    hubGroup
      .on("mouseenter", (ev, d) => {
        const bin = tokenBins().get(d.id);
        const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
        const yStr = years.length ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }` : "—";
        showTip(ev,
          `<aside><h6 class="bottom-padding">${ d.id }</h6>` +
          `Events: ${ d.eventCount }<br/>Connections: ${ d.hubDegree }<br/>` +
          `Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`);
      })
      .on("mousemove", moveTip).on("mouseleave", hideTip);

    nbGroup
      .on("mouseenter", (ev, d) => {
        const hubs = sharedByHubs();   // reactive read — fine inside D3 handler
        const lines = hubs.length
          ? hubs.slice(0, 5).map(h => `${ h.hub } (${ h.freq.toFixed(3) })`).join("<br/>")
          : "—";
        showTip(ev,
          `<aside><h6 class="bottom-padding">${ d.id }</h6>` +
          `Shared by ${ d.degree } hub(s):<br/>${ lines }</aside>`);
      })
      .on("mousemove", moveTip).on("mouseleave", hideTip);

    // ── Simulation ────────────────────────────────────────────────────────────
    if (simulationRef) simulationRef.stop();

    simulationRef = d3.forceSimulation<ContextNode>(nodes)
      .force("link",
        d3.forceLink<ContextNode, AnyEdge>(allEdges)
          .id(d => d.id)
          .distance(d =>
            d.kind === "hub-hub"
              ? Math.max(80, 260 - (d as HubHubEdge).weight * 180)
              : 60    // spokes: fixed short distance pulls nb nodes close to hub
          )
          .strength(d => d.kind === "hub-hub" ? 0.55 : 0.8)
      )
      .force("charge", d3.forceManyBody()
        .strength(d => d.kind === "hub" ? -280 : -40)
      )
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide<ContextNode>()
        .radius(d => d.kind === "hub" ? hubRadius(d.eventCount) + 8 : NB_RADIUS + 4)
      )
      .on("tick", () => {
        const x1 = (d: AnyEdge) => (d.source as ContextNode).x ?? 0;
        const y1 = (d: AnyEdge) => (d.source as ContextNode).y ?? 0;
        const x2 = (d: AnyEdge) => (d.target as ContextNode).x ?? 0;
        const y2 = (d: AnyEdge) => (d.target as ContextNode).y ?? 0;

        spokeSelection.attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2);
        hhSelection.attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2);

        hubHubEdges.forEach((d, i) =>
          defs.select(`#hh-${ i }`)
            .attr("x1", d.source.x ?? 0).attr("y1", d.source.y ?? 0)
            .attr("x2", d.target.x ?? 0).attr("y2", d.target.y ?? 0)
        );

        hubGroup.attr("transform", d => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`);
        nbGroup.attr("transform", d => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`);
      });
  }

  createEffect(() => { graphData(); if (svgRef) render(); });
  onCleanup(() => {
    simulationRef?.stop();
    d3.select("body").selectAll(".cg-tooltip").remove();
  });

  // ── UI ─────────────────────────────────────────────────────────────────────

  return (
    <>
      <style>{STYLES}</style>
      <div class="cg-layout">

        <header class="center-align fill max surface-container-low small-padding top-padding">
          <nav>

            <div class="field suffix border middle-align">
              <select value={concept()}
                onChange={(e) => { setConcept(e.currentTarget.value); setSelectedNode(null); }}>
                <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
              </select>
              <output>Concept</output>
            </div>

            <div class="field suffix border middle-align">
              <select value={maxHubs()}
                onChange={(e) => setMaxHubs(Number(e.currentTarget.value))}>
                <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
              </select>
              <output>Max hubs</output>
            </div>

            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={1} max={15} step={1} value={topN()}
                  onInput={(e) => setTopN(Number(e.currentTarget.value))} />
                <span /><span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">Top N {topN()}</output>
            </div>

            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={0.1} max={0.95} step={0.05} value={minSimilarity()}
                  onInput={(e) => setMinSimilarity(Number(e.currentTarget.value))} />
                <span /><span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                Min similarity {minSimilarity().toFixed(2)}
              </output>
            </div>

            <div class="field suffix border middle-align">
              <select value={yearMode()}
                onChange={(e) => setYearMode(e.currentTarget.value as "single" | "range")}>
                <option value="single">Single year</option>
                <option value="range">Year range</option>
              </select>
              <output>Year mode</output>
            </div>

            <Show when={yearMode() === "single"}>
              <div class="field middle-align">
                <div class="slider tiny">
                  <input type="range" min={CORPUS_START_YEAR} max={CORPUS_END_YEAR} step={1}
                    value={fromYear()}
                    onInput={(e) => { const v = Number(e.currentTarget.value); setFromYear(v); setToYear(v); }} />
                  <span class="tooltip bottom" />
                </div>
                <output class="small-padding top-padding">
                  {fromYear()} ({yearFiltered().length} events)
                </output>
              </div>
            </Show>

            <Show when={yearMode() === "range"}>
              <div class="field middle-align">
                <div class="slider tiny">
                  <input type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                    value={fromYear()}
                    onInput={(e) => setFromYear(Math.min(Number(e.currentTarget.value), toYear()))} />
                  <input type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                    value={toYear()}
                    onInput={(e) => setToYear(Math.max(Number(e.currentTarget.value), fromYear()))} />
                  <span /><span class="tooltip bottom" /><span class="tooltip bottom" />
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

        <div class="cg-main background">

          <svg ref={svgRef!} class="cg-svg surface-container-lowest" />

          <Show when={selectedNode()}>
            <aside class="cg-aside surface-container-high padding border">

              <div class="cg-header-row">
                <h2>{selectedNode()}</h2>
                <button class="link border" onClick={() => setSelectedNode(null)}>✕</button>
              </div>

              {/* ── Hub drill-down ── */}
              <Show when={selectedKind() === "hub" && selectedBin()}>
                {(_) => {
                  const bin = selectedBin()!;
                  const years = [...bin.years].sort((a, b) => a - b);
                  const topMax = bin.topNeighbours[0]?.freq ?? 1;
                  return (
                    <>
                      <div class="bottom-padding">
                        <div>Events: {bin.eventCount}</div>
                        <div>Documents: {bin.docs.size}</div>
                        <div>
                          Years:{" "}
                          {years.length
                            ? years.length === 1 ? years[0] : `${ years[0] }–${ years[years.length - 1] }`
                            : "—"}
                        </div>
                        <div>Hub connections: {graphData().nodes.find(n => n.id === selectedNode())?.hubDegree ?? 0}</div>
                      </div>

                      <h3 class="bottom-padding">Top neighbours</h3>
                      <div class="bottom-padding">
                        <For each={bin.topNeighbours.slice(0, 15)}>
                          {(nb) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div class="cg-nb-bar-fill hub"
                                  style={{ width: `${ (nb.freq / topMax) * 100 }%` }} />
                              </div>
                              <span class="cg-nb-token">{nb.token}</span>
                              <span class="cg-nb-score">{nb.meanScore.toFixed(3)}</span>
                            </div>
                          )}
                        </For>
                      </div>

                      <h3 class="bottom-padding">Sources</h3>
                      <Show when={selectedDocs().length > 0}
                        fallback={<div class="error">No documents found</div>}>
                        <For each={selectedDocs()}>
                          {([docId, pubYear]) => (
                            <button class="chip small-margin cg-chip-mono"
                              onClick={() => showDocument(docId)}>
                              <span>{docId}</span>
                              <Show when={pubYear !== undefined}>
                                <span class="small-text"> {pubYear}</span>
                              </Show>
                            </button>
                          )}
                        </For>
                      </Show>
                    </>
                  );
                }}
              </Show>

              {/* ── Neighbour drill-down ── */}
              <Show when={selectedKind() === "neighbour"}>
                <div class="bottom-padding">
                  <div>Shared by {sharedByHubs().length} hub(s)</div>
                </div>
                <h3 class="bottom-padding">Hub contexts</h3>
                <Show when={sharedByHubs().length > 0}
                  fallback={<div class="error">Not in any top-N list</div>}>
                  {(_) => {
                    const maxFreq = sharedByHubs()[0]?.freq ?? 1;
                    return (
                      <div class="bottom-padding">
                        <For each={sharedByHubs()}>
                          {(h) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div class="cg-nb-bar-fill neighbour"
                                  style={{ width: `${ (h.freq / maxFreq) * 100 }%` }} />
                              </div>
                              <span class="cg-nb-token">{h.hub}</span>
                              <span class="cg-nb-score">{h.meanScore.toFixed(3)}</span>
                            </div>
                          )}
                        </For>
                      </div>
                    );
                  }}
                </Show>
              </Show>

            </aside>
          </Show>

        </div>

        <footer class="fixed max center-align small-padding surface-container-low">
          <span class="cg-legend">
            <span class="cg-legend-hub" />hubs ({graphData().nodes.filter(n => n.kind === "hub").length})
            <span class="cg-legend-nb" />neighbours ({graphData().nodes.filter(n => n.kind === "neighbour").length})
            {" • "}{graphData().hubHubEdges.length} similarity edges
            {" • "}{graphData().hubNbEdges.length} spokes
            {" • "}{yearFiltered().length} events
            <Show when={fromYear() !== yearBounds()[0] || toYear() !== yearBounds()[1]}>
              {" • "}{fromYear()}–{toYear()}
            </Show>
          </span>
        </footer>

      </div>
    </>
  );
};

export default ContextGraph;
