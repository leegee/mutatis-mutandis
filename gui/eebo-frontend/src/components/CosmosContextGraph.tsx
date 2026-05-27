/**
 * ContextGraph6.tsx
 *
 * Port of ContextGraph5 from D3 force + SVG  →  cosmos.gl (v2.x, @cosmos.gl/graph).
 *
 * NOTE ON VERSIONS
 * ─────────────────
 * The README describes setConfigPartial() as a v3 feature.  v3 is currently a
 * GitHub-only beta (3.0.0-beta.6) and NOT yet published to npm.  The latest
 * published version is 2.6.x, which only has setConfig() — but that resets
 * ALL values to defaults before applying.  Effect 2 therefore re-passes every
 * relevant config property on each hubSpread change (see currentSimulation*
 * variables).  When you upgrade to v3, swap the setConfig() block in Effect 2
 * for a single setConfigPartial({ simulationRepulsion, simulationLinkSpring }).
 *
 * WHAT CHANGED vs ContextGraph5
 * ──────────────────────────────
 * • Rendering engine: D3 SVG tick loop → cosmos.gl GPU canvas
 * • Node terminology: D3 "node" → cosmos "point" (same objects, different words)
 * • Shapes: neighbour diamonds replaced by smaller amber circles (cosmos renders
 *   circles only; the diamond was purely cosmetic).
 * • Hub-hub gradients: replaced by per-link colour + width both scaled by weight.
 * • Dashed spokes: cosmos has no dashed-line support; spokes are thin + low alpha.
 * • Labels: HTML overlay div, hidden while simulating, shown on settle + hover.
 *   Uses cosmos trackPointPositionsByIndices / getTrackedPointPositionsMap.
 * • Drag: cosmos built-in (enableDrag: true).
 * • Zoom/pan: cosmos built-in.
 * • hubSpread: live-updates via setConfigPartial({ simulationRepulsion, ... })
 *   without rebuilding the graph — same contract as the original Effect 2.
 * • Tooltip: same HTML div pattern, positioned on onPointMouseOver.
 *
 * WHAT STAYED THE SAME
 * ─────────────────────
 * • All data functions (aggregateByToken, buildContextualGraph, buildPureEventGraph…)
 * • All SolidJS signals, memos, year filter, drill-down panel
 * • The two-effect pattern: Effect 1 rebuilds on graphData change,
 *   Effect 2 tweaks live forces on hubSpread change
 * • UI controls markup (untouched)
 *
 * COSMOS v3 QUICK PRIMER (read this if you've never used it)
 * ──────────────────────────────────────────────────────────
 * cosmos.gl represents graphs as parallel typed arrays:
 *
 *   pointPositions  Float32Array  [x0,y0, x1,y1, …]      (initial positions)
 *   pointColors     Float32Array  [r0,g0,b0,a0, …]        (0–1 per channel)
 *   pointSizes      Float32Array  [s0, s1, …]
 *   links           Float32Array  [src0,tgt0, src1,tgt1…] (indices into points)
 *   linkColors      Float32Array  [r0,g0,b0,a0, …]
 *   linkWidths      Float32Array  [w0, w1, …]
 *
 * You never touch SVG.  Cosmos owns a <canvas> inside the div you give it.
 *
 * Important v3 API notes:
 *   • new Graph(divElement, config)  — pass a DIV, not a canvas
 *   • graph.render()                 — call after every setPoint/setLink call
 *   • graph.setConfigPartial(…)      — partial update; setConfig() resets all defaults
 *   • graph.start()                  — restarts simulation (like alphaTarget restart)
 *   • onPointMouseOver(index, pos)   — index into your points array
 *   • onPointMouseOut()
 *   • onClick(index)
 *   • trackPointPositionsByIndices([…])  — registers indices to track
 *   • getTrackedPointPositionsMap()      — returns Map<number, [x,y]>
 *   • onSimulationEnd callback           — fires when alpha cools to rest
 *
 * NODE INDEX CONTRACT
 * ────────────────────
 * Cosmos knows points only by index, not by id.  We maintain:
 *   nodeIndexMap: Map<id, index>   built in buildCosmosArrays()
 *   indexToNode:  ContextNode[]    parallel array (index → node)
 * All event callbacks receive an index; we look up the node with indexToNode[i].
 */

import {
  createSignal,
  createMemo,
  createEffect,
  onCleanup,
  untrack,
  For,
  Show,
  type Component,
} from "solid-js";

import { Graph } from "@cosmos.gl/graph";
import * as d3 from "d3"; // kept for scale helpers only — no simulation
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

const MAX_TOP_N = 20;

// ─── Scoped styles (unchanged from ContextGraph5) ─────────────────────────────

const STYLES = `
  .cg-layout         { display:flex; flex-direction:column; height:100%; width:100%; }
  .cg-main           { display:flex; flex:1; overflow:hidden; position:relative; }
  .cg-canvas-wrap    { flex:1; position:relative; overflow:hidden; }
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
  .cg-legend-event   { display:inline-block; width:10px; height:10px;
                       border-radius:50%; background:rgba(120,210,130,.8);
                       flex-shrink:0; }
  .cg-legend-nb      { display:inline-block; width:9px; height:9px;
                       border-radius:50%; background:rgba(255,190,80,.85);
                       flex-shrink:0; }

  /* Label overlay */
  .cg-labels         { position:absolute; top:0; left:0; width:100%; height:100%;
                       pointer-events:none; overflow:hidden; }
  .cg-label          { position:absolute; transform:translate(-50%, 0);
                       white-space:nowrap;
                       font-family:'IBM Plex Mono','Courier New',monospace;
                       font-size:10px; pointer-events:none;
                       text-shadow: 0 1px 3px rgba(0,0,0,.8); }
  .cg-label.hub      { color:rgba(210,235,255,0.9); font-size:11px; }
  .cg-label.event    { color:rgba(180,255,190,0.8); font-size:9px; }
  .cg-label.neighbour{ color:rgba(255,220,140,0.85); font-size:10px; }
`;

// ─── Types (unchanged) ─────────────────────────────────────────────────────────

type ViewMode = "aggregated" | "events";

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
  neighbourScoreSum: Map<string, number>;
  topNeighbours: Array<{ token: string; freq: number; meanScore: number }>;
  docs: Map<string, number | undefined>;
  years: Set<number>;
}

// NOTE: we keep d3.SimulationNodeDatum off this interface — cosmos doesn't use
// x/y/vx/vy fields.  Position is managed entirely by the GPU.
interface ContextNode {
  id: string;
  kind: "hub" | "neighbour" | "event";
  eventCount: number;
  hubDegree: number;
  degree: number;
  token?: string;
  doc_id?: string;
  pub_year?: number;
}

interface HubHubEdge {
  kind: "hub-hub";
  sourceId: string;
  targetId: string;
  weight: number;
}

interface HubNbEdge {
  kind: "hub-neighbour";
  sourceId: string;
  targetId: string;
  weight: number;
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

// ─── Constants ────────────────────────────────────────────────────────────────

// All colours as [r,g,b,a] in 0-1 range (cosmos v3 requirement)
const HUB_COLOR_LOW_RGBA: [number, number, number, number] = [0.35, 0.53, 0.73, 0.40];
const HUB_COLOR_HIGH_RGBA: [number, number, number, number] = [0.91, 0.95, 0.99, 0.87];
const EVENT_RGBA: [number, number, number, number] = [0.47, 0.82, 0.51, 0.75];
const NB_RGBA: [number, number, number, number] = [1.00, 0.75, 0.31, 0.65];

// Hub-hub link colours (source/target sides for the "gradient" illusion):
// We can't do per-segment gradients in cosmos, so we just use a single mid-tone.
const HH_LINK_BASE_RGBA: [number, number, number, number] = [0.55, 0.75, 0.95, 0.95];
// Spoke links
const SPOKE_LINK_RGBA: [number, number, number, number] = [1.00, 0.75, 0.31, 0.90];

// Simulation tuning — kept close to original BASE_* constants
const BASE_REPULSION = 1.2;   // (0–2): how hard nodes push each other apart. hubSpread slider multiplies this.
const BASE_LINK_SPRING = 1;   // (0–2):  how tight edges pull nodes together
const BASE_FRICTION = 0.85;  // cosmos simulationFriction
const BASE_GRAVITY = 0.5;
const FIT_VIEW_PADDING = 1; //  graph spread/looks small, try 0.05. To zoom out try 0.3.
// const SPACE = 4096;
const SPACE = 2048;
const NB_POINT_SIZE = 8;
const HUB_MIN_SIZE = 12;
const HUB_MAX_SIZE = 40;
const EVENT_SIZE = 12;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [], hubHubEdges: [], hubNbEdges: [], allEdges: [],
  maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1,
};

// ─── Data functions (all unchanged from ContextGraph5) ─────────────────────────

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
  const bins = new Map<string, TokenBin>();

  for (const event of events) {
    const binKey = event.token;
    if (!binKey) continue;

    let bin = bins.get(binKey);
    if (!bin) {
      bin = {
        token: binKey, eventCount: 0,
        neighbourFreq: new Map(), neighbourScoreSum: new Map(),
        topNeighbours: [], docs: new Map(), years: new Set(),
      };
      bins.set(binKey, bin);
    }
    bin.eventCount += 1;
    if (event.doc_id) bin.docs.set(event.doc_id, event.pub_year);
    if (event.pub_year !== undefined) bin.years.add(event.pub_year);
    for (const nb of event.neighbours) {
      bin.neighbourFreq.set(nb.token, (bin.neighbourFreq.get(nb.token) ?? 0) + 1);
      bin.neighbourScoreSum.set(nb.token, (bin.neighbourScoreSum.get(nb.token) ?? 0) + nb.score);
    }
  }

  for (const bin of bins.values()) {
    const total = [...bin.neighbourFreq.values()].reduce((a, b) => a + b, 0);
    const normFreq = new Map<string, number>();
    for (const [tok, count] of bin.neighbourFreq)
      normFreq.set(tok, total > 0 ? count / total : 0);
    bin.neighbourFreq = normFreq;
    bin.topNeighbours = [...normFreq.entries()]
      .sort((a, b) => b[1] - a[1]).slice(0, 30)
      .map(([tok, freq]) => {
        const rawCount = Math.round(freq * total);
        return { token: tok, freq, meanScore: rawCount > 0 ? (bin.neighbourScoreSum.get(tok) ?? 0) / rawCount : 0 };
      });
  }
  return bins;
}

function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  let normA = 0; for (const v of a.values()) normA += v * v;
  let normB = 0; for (const v of b.values()) normB += v * v;
  normA = Math.sqrt(normA); normB = Math.sqrt(normB);
  if (normA === 0 || normB === 0) return 0;
  const [smaller, larger] = a.size <= b.size ? [a, b] : [b, a];
  let dot = 0;
  for (const [tok, v] of smaller) { const u = larger.get(tok); if (u !== undefined) dot += v * u; }
  return Math.min(1, Math.max(0, dot / (normA * normB)));
}

function buildContextualGraph(
  bins: Map<string, TokenBin>, topN: number, minSimilarity: number, maxHubs: number,
): ContextGraphData {
  const hubKeys = [...bins.keys()];
  if (hubKeys.length === 0) return EMPTY_GRAPH;

  const rawHubHub: Array<[string, string, number]> = [];
  for (let i = 0; i < hubKeys.length; i++)
    for (let j = i + 1; j < hubKeys.length; j++) {
      const sim = cosineSimilarity(bins.get(hubKeys[i])!.neighbourFreq, bins.get(hubKeys[j])!.neighbourFreq);
      if (sim >= minSimilarity) rawHubHub.push([hubKeys[i], hubKeys[j], sim]);
    }

  const hubHubDegree = new Map<string, number>();
  for (const [a, b] of rawHubHub) {
    hubHubDegree.set(a, (hubHubDegree.get(a) ?? 0) + 1);
    hubHubDegree.set(b, (hubHubDegree.get(b) ?? 0) + 1);
  }

  const sortedHubs = hubKeys
    .sort((a, b) => {
      const dd = (hubHubDegree.get(b) ?? 0) - (hubHubDegree.get(a) ?? 0);
      return dd !== 0 ? dd : bins.get(b)!.eventCount - bins.get(a)!.eventCount;
    }).slice(0, maxHubs);

  const hubSet = new Set(sortedHubs);
  const nodeMap = new Map<string, ContextNode>();
  for (const key of sortedHubs)
    nodeMap.set(key, { id: key, kind: "hub", eventCount: bins.get(key)!.eventCount, hubDegree: hubHubDegree.get(key) ?? 0, degree: hubHubDegree.get(key) ?? 0 });

  const spokeTriples: Array<[string, string, number]> = [];
  for (const hubKey of sortedHubs) {
    for (const nb of bins.get(hubKey)!.topNeighbours.slice(0, topN)) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, { id: nb.token, kind: "neighbour", eventCount: 0, hubDegree: 0, degree: 0 });
      spokeTriples.push([hubKey, nb.token, nb.freq]);
    }
  }
  for (const [hubKey, nbToken] of spokeTriples) {
    nodeMap.get(hubKey)!.degree += 1;
    nodeMap.get(nbToken)!.degree += 1;
  }

  const nodes = [...nodeMap.values()];
  const hubHubEdges: HubHubEdge[] = rawHubHub
    .filter(([a, b]) => hubSet.has(a) && hubSet.has(b))
    .map(([a, b, weight]) => ({ kind: "hub-hub" as const, sourceId: a, targetId: b, weight }));
  const hubNbEdges: HubNbEdge[] = spokeTriples.map(([s, t, w]) => ({ kind: "hub-neighbour" as const, sourceId: s, targetId: t, weight: w }));
  const allEdges: AnyEdge[] = [...hubNbEdges, ...hubHubEdges];

  const maxEventCount = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.eventCount));
  const maxHubDegree = Math.max(1, ...nodes.filter(n => n.kind === "hub").map(n => n.hubDegree));
  const maxHubHubWeight = Math.max(1, ...hubHubEdges.map(e => e.weight));

  console.log("[ctx-graph] hubs:", sortedHubs.length, "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length, "| hub-hub edges:", hubHubEdges.length, "| spoke edges:", hubNbEdges.length);
  return { nodes, hubHubEdges, hubNbEdges, allEdges, maxHubHubWeight, maxEventCount, maxHubDegree };
}

function buildPureEventGraph(events: ConceptEvent[], topN: number): ContextGraphData {
  if (events.length === 0) return EMPTY_GRAPH;
  const nodeMap = new Map<string, ContextNode>();
  const hubNbEdges: HubNbEdge[] = [];
  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const nodeId = event.event_id !== undefined ? `event_${ event.event_id }` : `event_idx:${ idx }`;
    const eventNode: ContextNode = { id: nodeId, kind: "event", eventCount: 1, hubDegree: 0, degree: 0, token: event.token, doc_id: event.doc_id, pub_year: event.pub_year };
    nodeMap.set(nodeId, eventNode);
    const top = [...event.neighbours].sort((a, b) => b.score - a.score).slice(0, topN);
    for (const nb of top) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, { id: nb.token, kind: "neighbour", eventCount: 0, hubDegree: 0, degree: 0 });
      hubNbEdges.push({ kind: "hub-neighbour" as const, sourceId: nodeId, targetId: nb.token, weight: nb.score });
      eventNode.degree += 1;
      nodeMap.get(nb.token)!.degree += 1;
    }
  }
  const nodes = [...nodeMap.values()];
  console.log("[ctx-graph/events] event nodes:", events.length, "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length, "| spoke edges:", hubNbEdges.length);
  return { nodes, hubHubEdges: [], hubNbEdges, allEdges: hubNbEdges, maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1 };
}

// ─── Cosmos typed-array builder ────────────────────────────────────────────────

interface CosmosArrays {
  pointPositions: Float32Array;  // [x,y, …] — random initial scatter
  pointColors: Float32Array;  // [r,g,b,a, …]
  pointSizes: Float32Array;  // [s, …]
  links: Float32Array;  // [src,tgt, …]
  linkColors: Float32Array;  // [r,g,b,a, …]
  linkWidths: Float32Array;  // [w, …]
  nodeIndexMap: Map<string, number>;  // id → point index
  indexToNode: ContextNode[];        // point index → node
}

function buildCosmosArrays(gd: ContextGraphData): CosmosArrays {
  const { nodes, allEdges, maxEventCount, maxHubDegree, maxHubHubWeight } = gd;
  const n = nodes.length;

  // D3 scale helpers (for colour interpolation only — no simulation)
  const hubColorScale = d3.scaleLinear<[number, number, number, number]>()
    .domain([0, Math.max(1, maxHubDegree)])
    .range([HUB_COLOR_LOW_RGBA, HUB_COLOR_HIGH_RGBA]);
  const hubSizeScale = d3.scaleSqrt().domain([0, maxEventCount]).range([HUB_MIN_SIZE, HUB_MAX_SIZE]);
  const hhAlphaScale = d3.scaleLinear().domain([0, maxHubHubWeight]).range([0.05, 0.25]);
  const hhWidthScale = d3.scaleLinear().domain([0, maxHubHubWeight]).range([0.5, 2.5]);
  const spokeAlpha = d3.scaleLinear().domain([0, 1]).range([0.15, 0.45]);
  const spokeWidth = d3.scaleLinear().domain([0, 1]).range([0.5, 2.5]);

  // Index map
  const nodeIndexMap = new Map<string, number>();
  const indexToNode: ContextNode[] = [];
  nodes.forEach((node, i) => { nodeIndexMap.set(node.id, i); indexToNode[i] = node; });

  // Point arrays
  const pointPositions = new Float32Array(n * 2);
  const pointColors = new Float32Array(n * 4);
  const pointSizes = new Float32Array(n);

  const cx = SPACE / 2, cy = SPACE / 2, spread = SPACE * 0.12;
  for (let i = 0; i < n; i++) {
    const node = nodes[i];
    // Random scatter within a circle, in [0, spaceSize] coordinates
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * spread;
    pointPositions[i * 2] = cx + Math.cos(angle) * r;
    pointPositions[i * 2 + 1] = cy + Math.sin(angle) * r;

    let rgba: [number, number, number, number];
    let size: number;
    if (node.kind === "hub") {
      rgba = hubColorScale(node.hubDegree);
      size = hubSizeScale(node.eventCount);
    } else if (node.kind === "event") {
      rgba = EVENT_RGBA;
      size = EVENT_SIZE;
    } else {
      rgba = NB_RGBA;
      size = NB_POINT_SIZE;
    }
    pointColors[i * 4] = rgba[0];
    pointColors[i * 4 + 1] = rgba[1];
    pointColors[i * 4 + 2] = rgba[2];
    pointColors[i * 4 + 3] = rgba[3];
    pointSizes[i] = size;
  }

  // Link arrays
  const m = allEdges.length;
  const links = new Float32Array(m * 2);
  const linkColors = new Float32Array(m * 4);
  const linkWidths = new Float32Array(m);

  for (let i = 0; i < m; i++) {
    const edge = allEdges[i];
    const si = nodeIndexMap.get(edge.sourceId) ?? 0;
    const ti = nodeIndexMap.get(edge.targetId) ?? 0;
    links[i * 2] = si;
    links[i * 2 + 1] = ti;

    if (edge.kind === "hub-hub") {
      const a = hhAlphaScale(edge.weight);
      const w = hhWidthScale(edge.weight);
      linkColors[i * 4] = HH_LINK_BASE_RGBA[0];
      linkColors[i * 4 + 1] = HH_LINK_BASE_RGBA[1];
      linkColors[i * 4 + 2] = HH_LINK_BASE_RGBA[2];
      linkColors[i * 4 + 3] = a;
      linkWidths[i] = w;
    } else {
      const a = spokeAlpha(edge.weight);
      const w = spokeWidth(edge.weight);
      linkColors[i * 4] = SPOKE_LINK_RGBA[0];
      linkColors[i * 4 + 1] = SPOKE_LINK_RGBA[1];
      linkColors[i * 4 + 2] = SPOKE_LINK_RGBA[2];
      linkColors[i * 4 + 3] = a;
      linkWidths[i] = w;
    }
  }

  return { pointPositions, pointColors, pointSizes, links, linkColors, linkWidths, nodeIndexMap, indexToNode };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

// ─── Component ────────────────────────────────────────────────────────────────

const ContextGraph6: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [viewMode, setViewMode] = createSignal<ViewMode>("aggregated");
  const [maxHubs, setMaxHubs] = createSignal(50);
  const [topN, setTopN] = createSignal(5);
  const [minSimilarity, setMinSimilarity] = createSignal(0.5);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);
  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);
  const [yearMode, setYearMode] = createSignal<"single" | "range">("single");
  const [hubSpread, setHubSpread] = createSignal(1);
  const [labelsVisible, setLabelsVisible] = createSignal(false);
  const [labelPositions, setLabelPositions] = createSignal<Array<{ id: string; kind: string; x: number; y: number }>>([]);
  const [hoveredId, setHoveredId] = createSignal<string | null>(null);

  // ── Year bounds (unchanged) ────────────────────────────────────────────────

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = props.data[concept()];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });

  createEffect(() => {
    const [min, max] = yearBounds();
    if (yearMode() === "single") { const mid = Math.floor((min + max) / 2); setFromYear(mid); setToYear(mid); }
    else { setFromYear(min); setToYear(max); }
  });

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    const events = cd.events;
    return fromYear() <= min && toYear() >= max ? events : filterByYearRange(events, fromYear(), toYear());
  });

  const tokenBins = createMemo<Map<string, TokenBin>>(() => aggregateByToken(yearFiltered()));

  const graphData = createMemo<ContextGraphData>(() =>
    viewMode() === "events"
      ? buildPureEventGraph(yearFiltered(), topN())
      : buildContextualGraph(tokenBins(), topN(), minSimilarity(), maxHubs())
  );

  // ── Drill-down (unchanged) ─────────────────────────────────────────────────

  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
    const id = selectedNode(); if (!id) return null;
    return graphData().nodes.find(n => n.id === id)?.kind ?? null;
  });
  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "hub") return null;
    return tokenBins().get(id) ?? null;
  });
  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const bin = selectedBin(); if (!bin) return [];
    return [...bin.docs.entries()].sort((a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity));
  });
  const selectedEventNode = createMemo<ContextNode | null>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find(n => n.id === id) ?? null;
  });
  const sharedByHubs = createMemo<Array<{ hub: string; freq: number; meanScore: number }>>(() => {
    const id = selectedNode(); if (!id || selectedKind() !== "neighbour") return [];
    if (viewMode() === "aggregated") {
      const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
      for (const [hubKey, bin] of tokenBins()) {
        const nb = bin.topNeighbours.find(n => n.token === id);
        if (nb) result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore });
      }
      return result.sort((a, b) => b.freq - a.freq);
    }
    const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
    for (const edge of graphData().hubNbEdges)
      if (edge.targetId === id)
        result.push({ hub: edge.sourceId, freq: edge.weight, meanScore: edge.weight });
    return result.sort((a, b) => b.freq - a.freq);
  });

  // ── Cosmos refs ───────────────────────────────────────────────────────────

  let wrapRef!: HTMLDivElement;
  let cosmosGraph: Graph | null = null;

  // Current index↔node mapping, kept in sync with each render() call
  let currentIndexToNode: ContextNode[] = [];
  let currentSimulationRepulsion = BASE_REPULSION;
  let currentSimulationLinkSpring = BASE_LINK_SPRING;

  // Tooltip element (same pattern as original)
  let tooltipEl: HTMLDivElement | null = null;
  const getTooltip = (): HTMLDivElement => {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "cg-tooltip surface-container-high border large-elevate padding";
      Object.assign(tooltipEl.style, { position: "fixed", pointerEvents: "none", fontFamily: "'IBM Plex Mono',monospace", opacity: "0", transition: "opacity 0.15s" });
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  };

  // ── Label overlay logic ───────────────────────────────────────────────────
  //
  // Strategy:
  //   • When simulation settles (onSimulationEnd) we track all point indices,
  //     read their canvas positions, project to screen coords, and render a
  //     div overlay.  Labels are hidden while moving (labelsVisible = false).
  //   • On hover we always show the hovered point's label regardless of settle.
  //   • Cosmos gives us canvas-space [x,y] from getTrackedPointPositionsMap().
  //     We then convert to CSS px using the wrapper's bounding rect.

  function updateLabelPositions() {
    if (!cosmosGraph) return;
    const posMap = cosmosGraph.getTrackedPointPositionsMap();
    if (!posMap || posMap.size === 0) return;
    const labels: Array<{ id: string; kind: string; x: number; y: number }> = [];
    posMap.forEach(([cx, cy], idx) => {
      const node = currentIndexToNode[idx];
      if (!node) return;
      // cosmos canvas coords are in [0, spaceSize] space; we need screen px.
      // The graph internally handles zoom/pan; getTrackedPointPositionsMap
      // returns SCREEN-space pixel coords relative to the canvas element.
      labels.push({ id: node.id, kind: node.kind, x: cx, y: cy });
    });
    setLabelPositions(labels);
  }

  // ── Core render (Effect 1 equivalent) ────────────────────────────────────
  //
  // Rebuilds cosmos arrays and calls graph.setPointPositions etc.
  // Never called for hubSpread changes (Effect 2 handles those).

  function renderGraph() {
    const gd = graphData();
    if (!wrapRef) return;

    // Initialise Graph lazily on first call
    if (!cosmosGraph) {
      cosmosGraph = new Graph(wrapRef, {
        backgroundColor: "#0a0c10",
        spaceSize: SPACE,
        enableDrag: true,
        fitViewPadding: FIT_VIEW_PADDING,
        simulationRepulsion: BASE_REPULSION,
        simulationLinkSpring: BASE_LINK_SPRING,
        simulationFriction: BASE_FRICTION,
        simulationGravity: BASE_GRAVITY,
        pointDefaultSize: NB_POINT_SIZE,
        linkDefaultWidth: 1,
        hoveredPointRingColor: "white",
        renderHoveredPointRing: true,

        onSimulationEnd: () => {
          console.debug("Simulation settled: read positions and show labels");
          updateLabelPositions();
          setLabelsVisible(true);
        },
        onSimulationStart: () => {
          console.debug("Simulation starting: hiding label overlay while nodes are in motion");
          setLabelsVisible(false);
        },

        onPointMouseOver: (index: number, pos: [number, number]) => {
          const node = currentIndexToNode[index];
          if (!node) return;
          setHoveredId(node.id);

          // Show single-label overlay for hovered node immediately
          setLabelPositions(prev => {
            // Ensure the hovered node has an up-to-date position
            const without = prev.filter(l => l.id !== node.id);
            without.push({ id: node.id, kind: node.kind, x: pos[0], y: pos[1] });
            return without;
          });

          // Tooltip
          const tip = getTooltip();
          let html = "";
          if (node.kind === "hub") {
            const bin = tokenBins().get(node.id);
            const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
            const yStr = years.length ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }` : "—";
            html = `<aside><h6 class="bottom-padding">${ node.id }</h6>Events: ${ node.eventCount }<br/>Connections: ${ node.hubDegree }<br/>Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`;
          } else if (node.kind === "event") {
            html = `<aside><h6 class="bottom-padding">${ node.token ?? node.id }</h6>Doc: ${ node.doc_id ?? "—" }<br/>Year: ${ node.pub_year ?? "—" }</aside>`;
          } else {
            const hubs = sharedByHubs();
            const lines = hubs.length ? hubs.slice(0, 5).map(h => `${ h.hub } (${ h.freq.toFixed(3) })`).join("<br/>") : "—";
            html = `<aside><h6 class="bottom-padding">${ node.id }</h6>Shared by ${ node.degree } source(s):<br/>${ lines }</aside>`;
          }
          tip.innerHTML = html;
          const [sx, sy] = pos;
          const bx = wrapRef.getBoundingClientRect();
          Object.assign(tip.style, { opacity: "1", left: `${ bx.left + sx + 14 }px`, top: `${ bx.top + sy - 10 }px` });
        },

        onPointMouseOut: () => {
          setHoveredId(null);
          getTooltip().style.opacity = "0";
        },

        onClick: (index: number | undefined) => {
          if (index === undefined) { setSelectedNode(null); return; }
          const node = currentIndexToNode[index];
          if (!node) return;
          setSelectedNode(prev => prev === node.id ? null : node.id);
        },
      });
    }

    if (gd.nodes.length === 0) {
      // Clear graph and show an empty state overlay (we can't draw text in cosmos)
      cosmosGraph.setPointPositions(new Float32Array(0));
      cosmosGraph.setLinks(new Float32Array(0));
      cosmosGraph.render();
      setLabelPositions([]);
      setLabelsVisible(false);
      return;
    }

    const arrays = buildCosmosArrays(gd);
    currentIndexToNode = arrays.indexToNode;

    // For degenerate graphs (single node, or all nodes isolated with no edges)
    // repulsion has nothing to balance against and nodes drift forever.
    // Pin them at the centre and cut repulsion to zero.
    const isDegenerate = gd.allEdges.length === 0 || gd.nodes.length <= 1;
    if (isDegenerate) {
      // Overwrite positions: place all nodes at centre
      for (let i = 0; i < gd.nodes.length; i++) {
        arrays.pointPositions[i * 2] = SPACE / 2;
        arrays.pointPositions[i * 2 + 1] = SPACE / 2;
      }
      cosmosGraph.setConfig({
        spaceSize: SPACE,
        simulationRepulsion: 0,
        simulationLinkSpring: 0,
        simulationFriction: BASE_FRICTION,
        simulationGravity: BASE_GRAVITY,
        enableDrag: true,
        fitViewPadding: FIT_VIEW_PADDING,
        hoveredPointRingColor: "white",
        renderHoveredPointRing: true,
      });
    } else {
      cosmosGraph.setConfig({
        spaceSize: SPACE,
        simulationRepulsion: BASE_REPULSION * (0.6 + hubSpread() * 0.4),
        simulationLinkSpring: BASE_LINK_SPRING,
        simulationFriction: BASE_FRICTION,
        simulationGravity: BASE_GRAVITY,
        enableDrag: true,
        fitViewPadding: FIT_VIEW_PADDING,
        hoveredPointRingColor: "white",
        renderHoveredPointRing: true,
      });
    }

    // Register all point indices for position tracking (needed for labels)
    cosmosGraph.trackPointPositionsByIndices(
      Array.from({ length: gd.nodes.length }, (_, i) => i)
    );

    cosmosGraph.setPointPositions(arrays.pointPositions);
    cosmosGraph.setPointColors(arrays.pointColors);
    cosmosGraph.setPointSizes(arrays.pointSizes);
    cosmosGraph.setLinks(arrays.links);
    cosmosGraph.setLinkColors(arrays.linkColors);
    cosmosGraph.setLinkWidths(arrays.linkWidths);
    cosmosGraph.render();
    // Fit the camera to the graph now that data is actually loaded.
    // A short delay lets the first simulation ticks spread nodes slightly
    // before the fit, so the view isn't a single-pixel dot.
    setTimeout(() => cosmosGraph?.fitView(), 300);

    setLabelsVisible(false); // labels will re-appear on next simulationEnd
  }

  // ── Effect 1: full rebuild when graphData changes ─────────────────────────
  //
  // hubSpread is NOT a dependency here.  untrack() guards any signal reads
  // inside renderGraph() (via closures) from registering as dependencies.

  createEffect(() => {
    graphData();
    if (wrapRef) untrack(() => renderGraph());
  });

  // ── Effect 2: live force tweak when hubSpread changes ────────────────────
  //
  // Maps the [0.2, 2.0] slider linearly onto simulationRepulsion.
  // We also bump simulationLinkDistance for the hub-hub distance analogue.

  createEffect(() => {
    const spread = hubSpread();
    if (!cosmosGraph) return;
    // Don't touch repulsion on degenerate graphs — it's already zeroed.
    const gd = untrack(graphData);
    if (gd.allEdges.length === 0 || gd.nodes.length <= 1) return;
    currentSimulationRepulsion = BASE_REPULSION * spread;
    currentSimulationLinkSpring = Math.min(0.92, BASE_FRICTION + (hubSpread() * 0.05));
    cosmosGraph.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: currentSimulationRepulsion,
      simulationLinkSpring: currentSimulationLinkSpring,
      simulationFriction: BASE_FRICTION,
      simulationGravity: BASE_GRAVITY,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white",
      renderHoveredPointRing: true,
    });
    // Nudge the simulation so changes are visible immediately
    cosmosGraph.start();
    setLabelsVisible(false);
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────

  onCleanup(() => {
    cosmosGraph?.pause();
    tooltipEl?.remove();
    tooltipEl = null;
  });

  // ── Render label overlay ──────────────────────────────────────────────────
  //
  // We show: all labels when labelsVisible=true, plus the hovered label always.

  const visibleLabels = createMemo(() => {
    const hid = hoveredId();
    const visible = labelsVisible();
    const positions = labelPositions();
    if (visible) return positions;
    // Only hovered label while moving
    if (hid) return positions.filter(l => l.id === hid);
    return [];
  });

  // ── UI (controls unchanged from ContextGraph5; only SVG → div) ───────────

  return (
    <>
      <style>{STYLES}</style>
      <div class="cg-layout">

        {/* ── Header controls (identical to ContextGraph5) ── */}
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
              <select value={viewMode()}
                onChange={(e) => { setViewMode(e.currentTarget.value as ViewMode); setSelectedNode(null); }}>
                <option value="aggregated">Aggregated</option>
                <option value="events">Events</option>
              </select>
              <output>View</output>
            </div>

            <Show when={viewMode() === "aggregated"}>
              <div class="field suffix border middle-align">
                <select value={maxHubs()}
                  onChange={(e) => setMaxHubs(Number(e.currentTarget.value))}>
                  <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
                </select>
                <output>Max hubs</output>
              </div>
            </Show>

            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={1} max={MAX_TOP_N} step={1} value={topN()}
                  onInput={(e) => setTopN(Number(e.currentTarget.value))} />
                <span /><span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">Top N {topN()}</output>
            </div>

            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={0.2} max={2.0} step={0.05} value={hubSpread()}
                  onInput={(e) => setHubSpread(Number(e.currentTarget.value))} />
                <span /><span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">Hub spread {hubSpread().toFixed(2)}</output>
            </div>

            <Show when={viewMode() === "aggregated"}>
              <div class="field middle-align">
                <div class="slider tiny">
                  <input type="range" min={0.01} max={0.95} step={0.05} value={minSimilarity()}
                    onInput={(e) => setMinSimilarity(Number(e.currentTarget.value))} />
                  <span /><span class="tooltip bottom" />
                </div>
                <output class="small-padding top-padding">Min sim {minSimilarity().toFixed(2)}</output>
              </div>
            </Show>

            <div class="field suffix border middle-align">
              <select value={yearMode()}
                onChange={(e) => setYearMode(e.currentTarget.value as "single" | "range")}>
                <option value="single">Single year</option>
                <option value="range">Year range</option>
              </select>
              <output>Year mode</output>
            </div>

            <Show when={yearMode() === "single"}>
              <nav class="no-space">
                <button class="circle chip secondary no-space large-margin bottom-margin"
                  onClick={() => { const v = Math.max(CORPUS_START_YEAR, fromYear() - 1); setFromYear(v); setToYear(v); }}>
                  <i>remove</i>
                </button>
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
                <button class="circle chip secondary no-space large-margin bottom-margin"
                  onClick={() => { const v = Math.min(CORPUS_END_YEAR, toYear() + 1); setToYear(v); setFromYear(v); }}>
                  <i>add</i>
                </button>
              </nav>
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
                  <span class="left-padding">{yearFiltered().length}/{props.data[concept()]?.n_events ?? 0} events</span>
                </output>
              </div>
            </Show>
          </nav>
        </header>

        {/* ── Main canvas + aside ── */}
        <div class="cg-main background">

          {/*
           * wrapRef is the div cosmos creates its canvas inside.
           * The .cg-labels overlay sits on top with pointer-events:none.
           */}
          <div ref={wrapRef!} class="cg-canvas-wrap surface-container-lowest">
            {/* Label overlay */}
            <div class="cg-labels">
              <For each={visibleLabels()}>
                {(lbl) => (
                  <span
                    class={`cg-label ${ lbl.kind }`}
                    style={{ left: `${ lbl.x }px`, top: `${ lbl.y + 14 }px` }}
                  >
                    {lbl.id}
                  </span>
                )}
              </For>
            </div>

            {/* Empty-state message (shown when graph has no nodes) */}
            <Show when={graphData().nodes.length === 0}>
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", "align-items": "center", "justify-content": "center",
                "pointer-events": "none",
              }}>
                <span class="error">
                  No graph: try reducing min similarity or increasing top N
                </span>
              </div>
            </Show>
          </div>

          {/* ── Drill-down aside (identical to ContextGraph5) ── */}
          <Show when={selectedNode()}>
            <aside class="cg-aside surface-container-high padding border">
              <div class="cg-header-row">
                <h2>{selectedNode()}</h2>
                <button class="link border" onClick={() => setSelectedNode(null)}>✕</button>
              </div>

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
                        <div>Years: {years.length ? (years.length === 1 ? years[0] : `${ years[0] }–${ years[years.length - 1] }`) : "—"}</div>
                        <div>Hub connections: {graphData().nodes.find(n => n.id === selectedNode())?.hubDegree ?? 0}</div>
                      </div>
                      <h3 class="bottom-padding">Top neighbours</h3>
                      <div class="bottom-padding">
                        <For each={bin.topNeighbours.slice(0, MAX_TOP_N)}>
                          {(nb) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div class="cg-nb-bar-fill hub" style={{ width: `${ (nb.freq / topMax) * 100 }%` }} />
                              </div>
                              <span class="cg-nb-token">{nb.token}</span>
                              <span class="cg-nb-score">{nb.meanScore.toFixed(3)}</span>
                            </div>
                          )}
                        </For>
                      </div>
                      <h3 class="bottom-padding">Sources</h3>
                      <Show when={selectedDocs().length > 0} fallback={<div class="error">No documents found</div>}>
                        <For each={selectedDocs()}>
                          {([docId, pubYear]) => (
                            <button class="chip small-margin cg-chip-mono" onClick={() => showDocument(docId)}>
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

              <Show when={selectedKind() === "event" && selectedEventNode()}>
                {(_) => {
                  const node = selectedEventNode()!;
                  return (
                    <>
                      <div class="bottom-padding">
                        <div>Token: {node.token ?? "—"}</div>
                        <div>Year: {node.pub_year ?? "—"}</div>
                        <Show when={node.doc_id}>
                          <div>
                            <button class="chip small-margin cg-chip-mono" onClick={() => showDocument(node.doc_id!)}>
                              <span>{node.doc_id}</span>
                            </button>
                          </div>
                        </Show>
                      </div>
                      <div class="bottom-padding small-text" style={{ opacity: 0.6 }}>
                        Select a neighbour to see which sources share it.
                      </div>
                    </>
                  );
                }}
              </Show>

              <Show when={selectedKind() === "neighbour"}>
                <div class="bottom-padding">
                  <div>Shared by {sharedByHubs().length} source(s)</div>
                </div>
                <h3 class="bottom-padding">
                  {viewMode() === "aggregated" ? "Hub contexts" : "Event contexts"}
                </h3>
                <Show when={sharedByHubs().length > 0} fallback={<div class="error">Not in any top-N list</div>}>
                  {(_) => {
                    const maxFreq = sharedByHubs()[0]?.freq ?? 1;
                    return (
                      <div class="bottom-padding">
                        <For each={sharedByHubs()}>
                          {(h) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div class="cg-nb-bar-fill neighbour" style={{ width: `${ (h.freq / maxFreq) * 100 }%` }} />
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

        {/* ── Footer legend (unchanged) ── */}
        <footer class="fixed max center-align small-padding surface-container-low">
          <span class="cg-legend">
            <Show when={viewMode() === "aggregated"}>
              <span class="cg-legend-hub" />hubs ({graphData().nodes.filter(n => n.kind === "hub").length})
            </Show>
            <Show when={viewMode() === "events"}>
              <span class="cg-legend-event" />events ({graphData().nodes.filter(n => n.kind === "event").length})
            </Show>
            <span class="cg-legend-nb" />neighbours ({graphData().nodes.filter(n => n.kind === "neighbour").length})
            <Show when={viewMode() === "aggregated"}>
              {" • "}{graphData().hubHubEdges.length} similarity edges
            </Show>
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

export default ContextGraph6;