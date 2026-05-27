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
import * as d3 from "d3";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

const MAX_TOP_N = 20;

// ─── Styles ────────────────────────────────────────────────────────────────────

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

  /* Label overlay — always-on, updated every rAF tick */
  .cg-labels         { position:absolute; top:0; left:0; width:100%; height:100%;
                       pointer-events:none; overflow:hidden; }
  .cg-label          { position:absolute; transform:translate(-50%, 0);
                       white-space:nowrap;
                       font-family:'IBM Plex Mono','Courier New',monospace;
                       font-size:10px; pointer-events:none;
                       text-shadow: 0 1px 3px rgba(0,0,0,.9),
                                    0 0   6px rgba(0,0,0,.7); }
  .cg-label.hub      { color:rgba(210,235,255,0.92); font-size:11px; font-weight:600; }
  .cg-label.event    { color:rgba(180,255,190,0.85); font-size:9px; }
  .cg-label.neighbour{ color:rgba(255,220,140,0.88); font-size:10px; }

  /* User guide panel */
  .cg-guide-toggle   { position:absolute; top:.6rem; right:.6rem; z-index:10;
                       opacity:.65; transition:opacity .15s; }
  .cg-guide-toggle:hover { opacity:1; }
  .cg-guide          { position:absolute; top:2.6rem; right:.6rem; z-index:10;
                       width:22rem; max-height:80vh; overflow-y:auto;
                       border-radius:.5rem;
                       font-size:.82rem; line-height:1.55; }
  .cg-guide h4       { margin:.6rem 0 .25rem; font-size:.85rem; opacity:.8; }
  .cg-guide p        { margin:.2rem 0 .5rem; opacity:.75; }
  .cg-guide ul       { margin:.2rem 0 .5rem; padding-left:1.1rem; opacity:.75; }
  .cg-guide li       { margin-bottom:.2rem; }
  .cg-guide-swatch   { display:inline-block; width:9px; height:9px;
                       border-radius:50%; vertical-align:middle;
                       margin-right:.3rem; flex-shrink:0; }

  /* Tooltip — positioned in VIEWPORT space */
  .cg-tooltip        { position:fixed; pointer-events:none;
                       font-family:'IBM Plex Mono',monospace;
                       opacity:0; transition:opacity .15s;
                       z-index:999; }
`;

// ─── Types ─────────────────────────────────────────────────────────────────────

type ViewMode = "aggregated" | "events";

interface Neighbour {
  token: string; score: number;
  event_id?: number; doc_id?: string; pub_year?: number; window_id?: number;
}
interface ConceptEvent {
  event_id?: number; token?: string; doc_id?: string; pub_year?: number;
  neighbours: Neighbour[];
}
interface ConceptData {
  n_events: number; year_min?: number; year_max?: number; events: ConceptEvent[];
}
export interface Tier2Data { [concept: string]: ConceptData; }
interface Props { data: Tier2Data; }

interface TokenBin {
  token: string; eventCount: number;
  neighbourFreq: Map<string, number>; neighbourScoreSum: Map<string, number>;
  topNeighbours: Array<{ token: string; freq: number; meanScore: number }>;
  docs: Map<string, number | undefined>; years: Set<number>;
}
interface ContextNode {
  id: string; kind: "hub" | "neighbour" | "event";
  eventCount: number; hubDegree: number; degree: number;
  token?: string; doc_id?: string; pub_year?: number;
}
interface HubHubEdge { kind: "hub-hub"; sourceId: string; targetId: string; weight: number; }
interface HubNbEdge { kind: "hub-neighbour"; sourceId: string; targetId: string; weight: number; }
type AnyEdge = HubHubEdge | HubNbEdge;
interface ContextGraphData {
  nodes: ContextNode[]; hubHubEdges: HubHubEdge[]; hubNbEdges: HubNbEdge[];
  allEdges: AnyEdge[]; maxHubHubWeight: number; maxEventCount: number; maxHubDegree: number;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const HUB_COLOR_LOW_RGBA: [number, number, number, number] = [0.35, 0.53, 0.73, 0.40];
const HUB_COLOR_HIGH_RGBA: [number, number, number, number] = [0.91, 0.95, 0.99, 0.87];
const EVENT_RGBA: [number, number, number, number] = [0.47, 0.82, 0.51, 0.75];
const NB_RGBA: [number, number, number, number] = [1.00, 0.75, 0.31, 0.65];
const HH_LINK_BASE_RGBA: [number, number, number, number] = [0.55, 0.75, 0.95, 0.95];
const SPOKE_LINK_RGBA: [number, number, number, number] = [1.00, 0.75, 0.31, 0.90];

// Simulation — friction is velocity *damping* (higher = stops faster).
// 0.85 was far too low (15% damping/tick = very slow cooldown).
// 0.4 gives noticeable damping without feeling sluggish.
const BASE_REPULSION = 1.2;
const BASE_LINK_SPRING = 1.0;
const BASE_FRICTION = 0.4;   // ← was 0.85 (now correctly high-damping)
const BASE_GRAVITY = 0.1;   // ← was 0.5 (less gravity → less fighting)
const FIT_VIEW_PADDING = 1;
const SPACE = window.innerHeight - (window.innerHeight / 6);
const NB_POINT_SIZE = 8;
const HUB_MIN_SIZE = 12;
const HUB_MAX_SIZE = 40;
const EVENT_SIZE = 12;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [], hubHubEdges: [], hubNbEdges: [], allEdges: [],
  maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1,
};

// ─── Data functions (unchanged) ───────────────────────────────────────────────

function scanYearRange(cd: ConceptData): [number, number] {
  let min = CORPUS_END_YEAR, max = CORPUS_START_YEAR;
  for (const e of cd.events) {
    if (e.pub_year === undefined) continue;
    if (e.pub_year < min) min = e.pub_year;
    if (e.pub_year > max) max = e.pub_year;
  }
  return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

function filterByYearRange(events: ConceptEvent[], from: number, to: number): ConceptEvent[] {
  return events.filter(e => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to);
}

function aggregateByToken(events: ConceptEvent[]): Map<string, TokenBin> {
  const bins = new Map<string, TokenBin>();
  for (const event of events) {
    const binKey = event.token;
    if (!binKey) continue;
    let bin = bins.get(binKey);
    if (!bin) {
      bin = { token: binKey, eventCount: 0, neighbourFreq: new Map(), neighbourScoreSum: new Map(), topNeighbours: [], docs: new Map(), years: new Set() };
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
    for (const [tok, count] of bin.neighbourFreq) normFreq.set(tok, total > 0 ? count / total : 0);
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
  for (const hubKey of sortedHubs)
    for (const nb of bins.get(hubKey)!.topNeighbours.slice(0, topN)) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, { id: nb.token, kind: "neighbour", eventCount: 0, hubDegree: 0, degree: 0 });
      spokeTriples.push([hubKey, nb.token, nb.freq]);
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
    for (const nb of [...event.neighbours].sort((a, b) => b.score - a.score).slice(0, topN)) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, { id: nb.token, kind: "neighbour", eventCount: 0, hubDegree: 0, degree: 0 });
      hubNbEdges.push({ kind: "hub-neighbour" as const, sourceId: nodeId, targetId: nb.token, weight: nb.score });
      eventNode.degree += 1;
      nodeMap.get(nb.token)!.degree += 1;
    }
  }
  const nodes = [...nodeMap.values()];
  return { nodes, hubHubEdges: [], hubNbEdges, allEdges: hubNbEdges, maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1 };
}

// ─── GraphWorld ────────────────────────────────────────────────────────────────
//
// Persistent simulation state — see previous version's header for full contract.
// This version adds a reverseIndex Map<cosmosIndex, WorldNode> to avoid O(n)
// scans in getNodeByIndex (called every rAF tick for labels + on every hover).

interface WorldNode extends ContextNode {
  cosmosIndex: number;
  cachedX: number;
  cachedY: number;
}
interface WorldEdge { key: string; edge: AnyEdge; cosmosRow: number; }
type DiffResult = {
  addedNodes: ContextNode[]; removedIds: string[]; updatedNodes: ContextNode[];
  edgesChanged: boolean; onlyVisuals: boolean;
};

function edgeKey(e: AnyEdge): string { return `${ e.sourceId }→${ e.targetId }`; }

class GraphWorld {
  private nodeRegistry = new Map<string, WorldNode>();
  private reverseIndex = new Map<number, WorldNode>();  // cosmosIndex → WorldNode
  private edgeRegistry = new Map<string, WorldEdge>();
  private indexPool: number[] = [];
  private nextIndex = 0;
  private totalSlots = 0;
  private cosmos: Graph;

  private hubColorScale!: d3.ScaleLinear<[number, number, number, number], [number, number, number, number]>;
  private hubSizeScale!: d3.ScalePower<number, number>;
  private hhAlphaScale!: d3.ScaleLinear<number, number>;
  private hhWidthScale!: d3.ScaleLinear<number, number>;
  private spokeAlpha!: d3.ScaleLinear<number, number>;
  private spokeWidth!: d3.ScaleLinear<number, number>;

  constructor(cosmos: Graph) { this.cosmos = cosmos; }

  applyDiff(gd: ContextGraphData): boolean {
    this.rebuildScales(gd);
    this.snapshotPositions();
    const diff = this.computeDiff(gd);

    for (const id of diff.removedIds) {
      const wn = this.nodeRegistry.get(id)!;
      this.reverseIndex.delete(wn.cosmosIndex);
      this.indexPool.push(wn.cosmosIndex);
      this.nodeRegistry.delete(id);
    }
    for (const cn of diff.addedNodes) {
      const idx = this.indexPool.length > 0 ? this.indexPool.pop()! : this.nextIndex++;
      this.totalSlots = Math.max(this.totalSlots, idx + 1);
      const [ix, iy] = this.initialPosition(cn, gd);
      const wn: WorldNode = { ...cn, cosmosIndex: idx, cachedX: ix, cachedY: iy };
      this.nodeRegistry.set(cn.id, wn);
      this.reverseIndex.set(idx, wn);
    }
    for (const cn of diff.updatedNodes) {
      const wn = this.nodeRegistry.get(cn.id)!;
      Object.assign(wn, cn);
    }
    if (diff.edgesChanged) {
      this.edgeRegistry.clear();
      gd.allEdges.forEach((edge, row) => this.edgeRegistry.set(edgeKey(edge), { key: edgeKey(edge), edge, cosmosRow: row }));
    }

    this.flushToCosmosArrays(gd.allEdges);
    this.cosmos.trackPointPositionsByIndices([...this.nodeRegistry.values()].map(wn => wn.cosmosIndex));

    if (!diff.onlyVisuals) {
      this.cosmos.start();
      return true;
    }
    return false;
  }

  reset() {
    this.nodeRegistry.clear();
    this.reverseIndex.clear();
    this.edgeRegistry.clear();
    this.indexPool = [];
    this.nextIndex = 0;
    this.totalSlots = 0;
    this.cosmos.setPointPositions(new Float32Array(0));
    this.cosmos.setLinks(new Float32Array(0));
    this.cosmos.render();
  }

  snapshotPositions() {
    const posMap = this.cosmos.getTrackedPointPositionsMap();
    if (!posMap) return;
    // getTrackedPointPositionsMap returns SCREEN-SPACE px relative to the
    // canvas element.  We store these as cachedX/Y so that when we flush
    // arrays after a diff, surviving nodes land at their last visible position.
    posMap.forEach(([x, y], idx) => {
      const wn = this.reverseIndex.get(idx);
      if (wn) { wn.cachedX = x; wn.cachedY = y; }
    });
  }

  // O(1) lookup — used on every rAF tick and on hover
  getNodeByIndex(idx: number): WorldNode | undefined {
    return this.reverseIndex.get(idx);
  }

  getNodeById(id: string): WorldNode | undefined {
    return this.nodeRegistry.get(id);
  }

  get liveNodes(): IterableIterator<WorldNode> {
    return this.nodeRegistry.values();
  }

  // ── Private ──────────────────────────────────────────────────────────────

  private rebuildScales(gd: ContextGraphData) {
    this.hubColorScale = d3.scaleLinear<[number, number, number, number]>()
      .domain([0, Math.max(1, gd.maxHubDegree)])
      .range([HUB_COLOR_LOW_RGBA, HUB_COLOR_HIGH_RGBA]);
    this.hubSizeScale = d3.scaleSqrt().domain([0, gd.maxEventCount]).range([HUB_MIN_SIZE, HUB_MAX_SIZE]);
    this.hhAlphaScale = d3.scaleLinear().domain([0, gd.maxHubHubWeight]).range([0.05, 0.25]);
    this.hhWidthScale = d3.scaleLinear().domain([0, gd.maxHubHubWeight]).range([0.5, 2.5]);
    this.spokeAlpha = d3.scaleLinear().domain([0, 1]).range([0.15, 0.45]);
    this.spokeWidth = d3.scaleLinear().domain([0, 1]).range([0.5, 2.5]);
  }

  private computeDiff(gd: ContextGraphData): DiffResult {
    const incomingIds = new Set(gd.nodes.map(n => n.id));
    const currentIds = new Set(this.nodeRegistry.keys());
    const addedNodes: ContextNode[] = [];
    const removedIds: string[] = [];
    const updatedNodes: ContextNode[] = [];

    for (const cn of gd.nodes) {
      if (!currentIds.has(cn.id)) {
        addedNodes.push(cn);
      } else {
        const wn = this.nodeRegistry.get(cn.id)!;
        if (wn.eventCount !== cn.eventCount || wn.hubDegree !== cn.hubDegree ||
          wn.degree !== cn.degree || wn.kind !== cn.kind)
          updatedNodes.push(cn);
      }
    }
    for (const id of currentIds) if (!incomingIds.has(id)) removedIds.push(id);

    const incomingEdgeKeys = new Set(gd.allEdges.map(edgeKey));
    const currentEdgeKeys = new Set(this.edgeRegistry.keys());
    const edgesChanged =
      incomingEdgeKeys.size !== currentEdgeKeys.size ||
      [...incomingEdgeKeys].some(k => !currentEdgeKeys.has(k));

    const onlyVisuals = addedNodes.length === 0 && removedIds.length === 0 && !edgesChanged;
    return { addedNodes, removedIds, updatedNodes, edgesChanged, onlyVisuals };
  }

  private initialPosition(cn: ContextNode, gd: ContextGraphData): [number, number] {
    const neighbourPositions: Array<[number, number]> = [];
    for (const edge of gd.allEdges) {
      let neighbourId: string | null = null;
      if (edge.sourceId === cn.id && this.nodeRegistry.has(edge.targetId))
        neighbourId = edge.targetId;
      else if (edge.targetId === cn.id && this.nodeRegistry.has(edge.sourceId))
        neighbourId = edge.sourceId;
      if (neighbourId) {
        const wn = this.nodeRegistry.get(neighbourId)!;
        neighbourPositions.push([wn.cachedX, wn.cachedY]);
      }
    }
    const jitter = () => (Math.random() - 0.5) * SPACE * 0.04;
    if (neighbourPositions.length > 0) {
      const cx = neighbourPositions.reduce((s, p) => s + p[0], 0) / neighbourPositions.length;
      const cy = neighbourPositions.reduce((s, p) => s + p[1], 0) / neighbourPositions.length;
      return [cx + jitter(), cy + jitter()];
    }
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * SPACE * 0.08;
    return [SPACE / 2 + Math.cos(angle) * r, SPACE / 2 + Math.sin(angle) * r];
  }

  private flushToCosmosArrays(edges: AnyEdge[]) {
    const slots = this.totalSlots;
    const pointPositions = new Float32Array(slots * 2);
    const pointColors = new Float32Array(slots * 4);
    const pointSizes = new Float32Array(slots);

    for (const wn of this.nodeRegistry.values()) {
      const i = wn.cosmosIndex;
      pointPositions[i * 2] = wn.cachedX;
      pointPositions[i * 2 + 1] = wn.cachedY;
      let rgba: [number, number, number, number];
      let size: number;
      if (wn.kind === "hub") { rgba = this.hubColorScale(wn.hubDegree); size = this.hubSizeScale(wn.eventCount); }
      else if (wn.kind === "event") { rgba = EVENT_RGBA; size = EVENT_SIZE; }
      else { rgba = NB_RGBA; size = NB_POINT_SIZE; }
      pointColors[i * 4] = rgba[0]; pointColors[i * 4 + 1] = rgba[1];
      pointColors[i * 4 + 2] = rgba[2]; pointColors[i * 4 + 3] = rgba[3];
      pointSizes[i] = size;
    }

    const m = edges.length;
    const links = new Float32Array(m * 2);
    const linkColors = new Float32Array(m * 4);
    const linkWidths = new Float32Array(m);
    for (let i = 0; i < m; i++) {
      const edge = edges[i];
      links[i * 2] = this.nodeRegistry.get(edge.sourceId)?.cosmosIndex ?? 0;
      links[i * 2 + 1] = this.nodeRegistry.get(edge.targetId)?.cosmosIndex ?? 0;
      if (edge.kind === "hub-hub") {
        const a = this.hhAlphaScale(edge.weight), w = this.hhWidthScale(edge.weight);
        linkColors[i * 4] = HH_LINK_BASE_RGBA[0]; linkColors[i * 4 + 1] = HH_LINK_BASE_RGBA[1];
        linkColors[i * 4 + 2] = HH_LINK_BASE_RGBA[2]; linkColors[i * 4 + 3] = a;
        linkWidths[i] = w;
      } else {
        const a = this.spokeAlpha(edge.weight), w = this.spokeWidth(edge.weight);
        linkColors[i * 4] = SPOKE_LINK_RGBA[0]; linkColors[i * 4 + 1] = SPOKE_LINK_RGBA[1];
        linkColors[i * 4 + 2] = SPOKE_LINK_RGBA[2]; linkColors[i * 4 + 3] = a;
        linkWidths[i] = w;
      }
    }

    this.cosmos.setPointPositions(pointPositions);
    this.cosmos.setPointColors(pointColors);
    this.cosmos.setPointSizes(pointSizes);
    this.cosmos.setLinks(links);
    this.cosmos.setLinkColors(linkColors);
    this.cosmos.setLinkWidths(linkWidths);
    this.cosmos.render();
  }
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
  const [labelPositions, setLabelPositions] = createSignal<Array<{ id: string; kind: string; x: number; y: number }>>([]);
  const [hoveredId, setHoveredId] = createSignal<string | null>(null);

  // ── Year bounds ──────────────────────────────────────────────────────────

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
    return fromYear() <= min && toYear() >= max
      ? cd.events
      : filterByYearRange(cd.events, fromYear(), toYear());
  });

  const tokenBins = createMemo<Map<string, TokenBin>>(() => aggregateByToken(yearFiltered()));

  const graphData = createMemo<ContextGraphData>(() =>
    viewMode() === "events"
      ? buildPureEventGraph(yearFiltered(), topN())
      : buildContextualGraph(tokenBins(), topN(), minSimilarity(), maxHubs())
  );

  // ── Drill-down ───────────────────────────────────────────────────────────

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
    return graphData().hubNbEdges
      .filter(e => e.targetId === id)
      .map(e => ({ hub: e.sourceId, freq: e.weight, meanScore: e.weight }))
      .sort((a, b) => b.freq - a.freq);
  });

  // ── Cosmos + GraphWorld refs ─────────────────────────────────────────────

  let wrapRef!: HTMLDivElement;
  let cosmosGraph: Graph | null = null;
  let world: GraphWorld | null = null;
  let rafHandle = 0;

  // ── Tooltip ──────────────────────────────────────────────────────────────
  //
  // FIX: cosmos onPointMouseOver fires with pos = [clientX, clientY] —
  // already in VIEWPORT space.  The previous code added bx.left/top on top,
  // double-counting the canvas offset and pushing the tooltip far off.
  // Now we use pos directly as the fixed viewport coordinates.

  let tooltipEl: HTMLDivElement | null = null;
  const getTooltip = (): HTMLDivElement => {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "cg-tooltip surface-container-high border large-elevate padding";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  };

  function showTooltip(wn: WorldNode, pos: [number, number]) {
    const tip = getTooltip();
    let html = "";
    if (wn.kind === "hub") {
      const bin = tokenBins().get(wn.id);
      const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
      const yStr = years.length
        ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }`
        : "—";
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Events: ${ wn.eventCount }<br/>Connections: ${ wn.hubDegree }<br/>Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`;
    } else if (wn.kind === "event") {
      html = `<aside><h6 class="bottom-padding">${ wn.token ?? wn.id }</h6>Doc: ${ wn.doc_id ?? "—" }<br/>Year: ${ wn.pub_year ?? "—" }</aside>`;
    } else {
      const hubs = sharedByHubs();
      const lines = hubs.length
        ? hubs.slice(0, 5).map(h => `${ h.hub } (${ h.freq.toFixed(3) })`).join("<br/>")
        : "—";
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Shared by ${ wn.degree } source(s):<br/>${ lines }</aside>`;
    }
    tip.innerHTML = html;
    // pos is viewport-relative — use directly as fixed position + small offset
    Object.assign(tip.style, {
      opacity: "1",
      left: `${ pos[0] + 14 }px`,
      top: `${ pos[1] - 10 }px`,
    });
  }

  // ── Label rAF loop ────────────────────────────────────────────────────────
  //
  // FIX: Rather than waiting for onSimulationEnd (which is fragile — it fires
  // once then simulation restarts on every applyDiff render() call), we run a
  // rAF loop that reads positions continuously.  Labels are always shown for
  // nodes whose id !== concept (neighbours / non-concept hubs / events).
  // The concept hubs are numerous and overlap badly, so we suppress them by
  // default; only the hovered node's label is shown while it is hovered.
  //
  // getTrackedPointPositionsMap() returns canvas-element-relative px (not world
  // space), so they map directly onto the .cg-labels overlay div.

  function startLabelLoop() {
    cancelAnimationFrame(rafHandle);
    function tick() {
      if (!cosmosGraph || !world) { rafHandle = requestAnimationFrame(tick); return; }
      const posMap = cosmosGraph.getTrackedPointPositionsMap();
      if (posMap && posMap.size > 0) {
        const hid = hoveredId();
        const labels: Array<{ id: string; kind: string; x: number; y: number }> = [];
        const currentConcept = untrack(concept);
        posMap.forEach(([x, y], idx) => {
          const wn = world!.getNodeByIndex(idx);
          if (!wn) return;
          // Show label for: non-concept nodes always; hovered node always
          const isConceptHub = wn.kind === "hub" && wn.id === currentConcept;
          if (!isConceptHub || wn.id === hid)
            labels.push({ id: wn.id, kind: wn.kind, x, y });
        });
        setLabelPositions(labels);
      }
      rafHandle = requestAnimationFrame(tick);
    }
    rafHandle = requestAnimationFrame(tick);
  }

  // ── Initialise cosmos (once) ─────────────────────────────────────────────

  function ensureCosmosInitialised() {
    if (cosmosGraph) return;

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

      onPointMouseOver: (index: number, pos: [number, number]) => {
        const wn = world?.getNodeByIndex(index);
        if (!wn) return;
        setHoveredId(wn.id);
        showTooltip(wn, pos);
      },
      onPointMouseOut: () => {
        setHoveredId(null);
        if (tooltipEl) tooltipEl.style.opacity = "0";
      },
      onClick: (index: number | undefined) => {
        if (index === undefined) { setSelectedNode(null); return; }
        const wn = world?.getNodeByIndex(index);
        if (!wn) return;
        setSelectedNode(prev => prev === wn.id ? null : wn.id);
      },
    });

    world = new GraphWorld(cosmosGraph);
    startLabelLoop();
  }

  // ── Effect 1: diff on graphData change ───────────────────────────────────

  let lastConcept = "";

  createEffect(() => {
    const gd = graphData();
    const current = untrack(concept);

    if (!wrapRef) return;
    ensureCosmosInitialised();

    if (current !== lastConcept) {
      world!.reset();
      lastConcept = current;
    }

    if (gd.nodes.length === 0) {
      world!.reset();
      setLabelPositions([]);
      return;
    }

    const isDegenerate = gd.allEdges.length === 0 || gd.nodes.length <= 1;

    cosmosGraph!.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: isDegenerate ? 0 : BASE_REPULSION * (0.6 + untrack(hubSpread) * 0.4),
      simulationLinkSpring: isDegenerate ? 0 : BASE_LINK_SPRING,
      simulationFriction: BASE_FRICTION,
      simulationGravity: BASE_GRAVITY,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white",
      renderHoveredPointRing: true,
    });

    const topologyChanged = world!.applyDiff(gd);
    if (topologyChanged) setTimeout(() => cosmosGraph?.fitView(), 300);
  });

  // ── Effect 2: hubSpread force tweak ──────────────────────────────────────

  createEffect(() => {
    const spread = hubSpread();
    if (!cosmosGraph || !world) return;
    const gd = untrack(graphData);
    if (gd.allEdges.length === 0 || gd.nodes.length <= 1) return;
    cosmosGraph.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: BASE_REPULSION * spread,
      simulationLinkSpring: Math.min(0.92, BASE_FRICTION + spread * 0.05),
      simulationFriction: BASE_FRICTION,
      simulationGravity: BASE_GRAVITY,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white",
      renderHoveredPointRing: true,
    });
    cosmosGraph.start();
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────

  onCleanup(() => {
    cancelAnimationFrame(rafHandle);
    cosmosGraph?.pause();
    tooltipEl?.remove();
    tooltipEl = null;
  });

  // ── UI ────────────────────────────────────────────────────────────────────

  return (
    <>
      <style>{STYLES}</style>
      <div class="cg-layout">

        {/* ── Header controls ── */}
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

          <div ref={wrapRef!} class="cg-canvas-wrap surface-container-lowest">

            {/* Label overlay — updated every rAF tick, always visible */}
            <div class="cg-labels">
              <For each={labelPositions()}>
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

            {/* Empty-state */}
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

          {/* ── Drill-down aside ── */}
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

        {/* ── Footer legend ── */}
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