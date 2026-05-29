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

import "./styles.css";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../../corpus_config";

import type {
  AnyEdge,
  ConceptData,
  ConceptEvent,
  ContextGraphData,
  ContextNode,
  HubHubEdge,
  HubNbEdge,
  TokenBin,
} from "./types";
import { controls, setControls } from "../../state/controls.store";
import ControlsHeader from "../ControlsHeader";
import { tier2Data } from "../../state/tier2data.store";

const MAX_TOP_N = 20;

const HUB_COLOR_LOW_RGBA: [number, number, number, number] = [0.35, 0.53, 0.73, 0.4];
const HUB_COLOR_HIGH_RGBA: [number, number, number, number] = [0.91, 0.95, 0.99, 0.87];
const EVENT_RGBA: [number, number, number, number] = [0.47, 0.82, 0.51, 0.75];
const NB_RGBA: [number, number, number, number] = [1.0, 0.75, 0.31, 0.65];
const HH_LINK_BASE_RGBA: [number, number, number, number] = [0.55, 0.75, 0.95, 0.95];
const SPOKE_LINK_RGBA: [number, number, number, number] = [1.0, 0.75, 0.31, 0.9];

const BASE_REPULSION = 0.8;
const BASE_LINK_SPRING = 1.0;
const BASE_FRICTION = 0.4;
const BASE_GRAVITY = 0.25;
const FIT_VIEW_PADDING = 1;
const SPACE = window.innerHeight - window.innerHeight / 6;
const NB_POINT_SIZE = 8;
const HUB_MIN_SIZE = 12;
const HUB_MAX_SIZE = 40;
const EVENT_SIZE = 12;

// How many px a node must move before we bother updating labels.
const LABEL_MOVE_THRESHOLD = 0.5;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [],
  hubHubEdges: [],
  hubNbEdges: [],
  allEdges: [],
  maxHubHubWeight: 1,
  maxEventCount: 1,
  maxHubDegree: 1,
};

// ─── Data functions ────────────────────────────────────────────────────────────

function scanYearRange(cd: ConceptData): [number, number] {
  let min = CORPUS_END_YEAR,
    max = CORPUS_START_YEAR;
  for (const e of cd.events) {
    if (e.pub_year === undefined) continue;
    if (e.pub_year < min) min = e.pub_year;
    if (e.pub_year > max) max = e.pub_year;
  }
  return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

function filterByYearRange(
  events: ConceptEvent[],
  from: number,
  to: number
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
        token: binKey,
        eventCount: 0,
        neighbourFreq: new Map(),
        neighbourScoreSum: new Map(),
        topNeighbours: [],
        docs: new Map(),
        years: new Set(),
      };
      bins.set(binKey, bin);
    }
    bin.eventCount += 1;
    if (event.doc_id) bin.docs.set(event.doc_id, event.pub_year);
    if (event.pub_year !== undefined) bin.years.add(event.pub_year);
    for (const nb of event.neighbours) {
      bin.neighbourFreq.set(nb.token, (bin.neighbourFreq.get(nb.token) ?? 0) + 1);
      bin.neighbourScoreSum.set(
        nb.token,
        (bin.neighbourScoreSum.get(nb.token) ?? 0) + nb.score
      );
    }
  }
  for (const bin of bins.values()) {
    const total = [...bin.neighbourFreq.values()].reduce((a, b) => a + b, 0);
    const normFreq = new Map<string, number>();
    for (const [tok, count] of bin.neighbourFreq)
      normFreq.set(tok, total > 0 ? count / total : 0);
    bin.neighbourFreq = normFreq;
    bin.topNeighbours = [...normFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)
      .map(([tok, freq]) => {
        const rawCount = Math.round(freq * total);
        return {
          token: tok,
          freq,
          meanScore:
            rawCount > 0 ? (bin.neighbourScoreSum.get(tok) ?? 0) / rawCount : 0,
        };
      });
  }
  return bins;
}

function cosineSimilarity(
  a: Map<string, number>,
  b: Map<string, number>
): number {
  let normA = 0;
  for (const v of a.values()) normA += v * v;
  let normB = 0;
  for (const v of b.values()) normB += v * v;
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

function buildContextualGraph(
  bins: Map<string, TokenBin>,
  topN: number,
  minSimilarity: number,
  maxHubs: number
): ContextGraphData {
  const hubKeys = [...bins.keys()];
  if (hubKeys.length === 0) return EMPTY_GRAPH;
  const rawHubHub: Array<[string, string, number]> = [];
  for (let i = 0; i < hubKeys.length; i++)
    for (let j = i + 1; j < hubKeys.length; j++) {
      const sim = cosineSimilarity(
        bins.get(hubKeys[i])!.neighbourFreq,
        bins.get(hubKeys[j])!.neighbourFreq
      );
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
    })
    .slice(0, maxHubs);
  const hubSet = new Set(sortedHubs);
  const nodeMap = new Map<string, ContextNode>();
  for (const key of sortedHubs)
    nodeMap.set(key, {
      id: key,
      kind: "hub",
      eventCount: bins.get(key)!.eventCount,
      hubDegree: hubHubDegree.get(key) ?? 0,
      degree: hubHubDegree.get(key) ?? 0,
    });
  const spokeTriples: Array<[string, string, number]> = [];
  for (const hubKey of sortedHubs)
    for (const nb of bins.get(hubKey)!.topNeighbours.slice(0, topN)) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, {
          id: nb.token,
          kind: "neighbour",
          eventCount: 0,
          hubDegree: 0,
          degree: 0,
        });
      spokeTriples.push([hubKey, nb.token, nb.freq]);
    }
  for (const [hubKey, nbToken] of spokeTriples) {
    nodeMap.get(hubKey)!.degree += 1;
    nodeMap.get(nbToken)!.degree += 1;
  }
  const nodes = [...nodeMap.values()];
  const hubHubEdges: HubHubEdge[] = rawHubHub
    .filter(([a, b]) => hubSet.has(a) && hubSet.has(b))
    .map(([a, b, weight]) => ({
      kind: "hub-hub" as const,
      sourceId: a,
      targetId: b,
      weight,
    }));
  const hubNbEdges: HubNbEdge[] = spokeTriples.map(([s, t, w]) => ({
    kind: "hub-neighbour" as const,
    sourceId: s,
    targetId: t,
    weight: w,
  }));
  const allEdges: AnyEdge[] = [...hubNbEdges, ...hubHubEdges];
  const maxEventCount = Math.max(
    1,
    ...nodes.filter((n) => n.kind === "hub").map((n) => n.eventCount)
  );
  const maxHubDegree = Math.max(
    1,
    ...nodes.filter((n) => n.kind === "hub").map((n) => n.hubDegree)
  );
  const maxHubHubWeight = Math.max(1, ...hubHubEdges.map((e) => e.weight));
  return {
    nodes,
    hubHubEdges,
    hubNbEdges,
    allEdges,
    maxHubHubWeight,
    maxEventCount,
    maxHubDegree,
  };
}

function buildPureEventGraph(
  events: ConceptEvent[],
  topN: number
): ContextGraphData {
  if (events.length === 0) return EMPTY_GRAPH;
  const nodeMap = new Map<string, ContextNode>();
  const hubNbEdges: HubNbEdge[] = [];
  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const nodeId =
      event.event_id !== undefined
        ? `event_${ event.event_id }`
        : `event_idx:${ idx }`;
    const eventNode: ContextNode = {
      id: nodeId,
      kind: "event",
      eventCount: 1,
      hubDegree: 0,
      degree: 0,
      token: event.token,
      doc_id: event.doc_id,
      pub_year: event.pub_year,
    };
    nodeMap.set(nodeId, eventNode);
    for (const nb of [...event.neighbours]
      .sort((a, b) => b.score - a.score)
      .slice(0, topN)) {
      if (!nodeMap.has(nb.token))
        nodeMap.set(nb.token, {
          id: nb.token,
          kind: "neighbour",
          eventCount: 0,
          hubDegree: 0,
          degree: 0,
        });
      hubNbEdges.push({
        kind: "hub-neighbour" as const,
        sourceId: nodeId,
        targetId: nb.token,
        weight: nb.score,
      });
      eventNode.degree += 1;
      nodeMap.get(nb.token)!.degree += 1;
    }
  }
  const nodes = [...nodeMap.values()];
  return {
    nodes,
    hubHubEdges: [],
    hubNbEdges,
    allEdges: hubNbEdges,
    maxHubHubWeight: 1,
    maxEventCount: 1,
    maxHubDegree: 1,
  };
}

// ─── GraphWorld ────────────────────────────────────────────────────────────────

interface WorldNode extends ContextNode {
  cosmosIndex: number;
  cachedX: number;
  cachedY: number;
}
interface WorldEdge {
  key: string;
  edge: AnyEdge;
  cosmosRow: number;
}
type DiffResult = {
  addedNodes: ContextNode[];
  removedIds: string[];
  updatedNodes: ContextNode[];
  edgesChanged: boolean;
  onlyVisuals: boolean;
};

function edgeKey(e: AnyEdge): string {
  return `${ e.sourceId }→${ e.targetId }`;
}

class GraphWorld {
  private nodeRegistry = new Map<string, WorldNode>();
  private reverseIndex = new Map<number, WorldNode>();
  private edgeRegistry = new Map<string, WorldEdge>();
  private indexPool: number[] = [];
  private nextIndex = 0;
  private totalSlots = 0;
  private cosmos: Graph;

  // ── Pre-allocated TypedArrays — grown only when slot count exceeds capacity.
  // Reusing the same buffer avoids GC churn on every topology/visual update.
  private _pointPositions = new Float32Array(0);
  private _pointColors = new Float32Array(0);
  private _pointSizes = new Float32Array(0);
  private _links = new Float32Array(0);
  private _linkColors = new Float32Array(0);
  private _linkWidths = new Float32Array(0);

  private hubColorScale!: d3.ScaleLinear<
    [number, number, number, number],
    [number, number, number, number]
  >;
  private hubSizeScale!: d3.ScalePower<number, number>;
  private hhAlphaScale!: d3.ScaleLinear<number, number>;
  private hhWidthScale!: d3.ScaleLinear<number, number>;
  private spokeAlpha!: d3.ScaleLinear<number, number>;
  private spokeWidth!: d3.ScaleLinear<number, number>;

  constructor(cosmos: Graph) {
    this.cosmos = cosmos;
  }

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

    // Build an adjacency index once — used by initialPosition to avoid
    // scanning the full edge list for every new node (was O(nodes × edges)).
    const adjIndex =
      diff.addedNodes.length > 0
        ? this.buildAdjacencyIndex(gd.allEdges)
        : null;

    for (const cn of diff.addedNodes) {
      const idx =
        this.indexPool.length > 0 ? this.indexPool.pop()! : this.nextIndex++;
      this.totalSlots = Math.max(this.totalSlots, idx + 1);
      const [ix, iy] = this.initialPosition(cn, gd, adjIndex!);
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
      gd.allEdges.forEach((edge, row) =>
        this.edgeRegistry.set(edgeKey(edge), { key: edgeKey(edge), edge, cosmosRow: row })
      );
    }

    this.flushToCosmosArrays(gd.allEdges);
    this.cosmos.trackPointPositionsByIndices(
      [...this.nodeRegistry.values()].map((wn) => wn.cosmosIndex)
    );

    return !diff.onlyVisuals;
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
    posMap.forEach(([x, y], idx) => {
      const wn = this.reverseIndex.get(idx);
      if (wn) {
        wn.cachedX = x;
        wn.cachedY = y;
      }
    });
  }

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
    this.hubColorScale = d3
      .scaleLinear<[number, number, number, number]>()
      .domain([0, Math.max(1, gd.maxHubDegree)])
      .range([HUB_COLOR_LOW_RGBA, HUB_COLOR_HIGH_RGBA]);
    this.hubSizeScale = d3
      .scaleSqrt()
      .domain([0, gd.maxEventCount])
      .range([HUB_MIN_SIZE, HUB_MAX_SIZE]);
    this.hhAlphaScale = d3
      .scaleLinear()
      .domain([0, gd.maxHubHubWeight])
      .range([0.05, 0.25]);
    this.hhWidthScale = d3
      .scaleLinear()
      .domain([0, gd.maxHubHubWeight])
      .range([0.5, 2.5]);
    this.spokeAlpha = d3.scaleLinear().domain([0, 1]).range([0.15, 0.45]);
    this.spokeWidth = d3.scaleLinear().domain([0, 1]).range([0.5, 2.5]);
  }

  private computeDiff(gd: ContextGraphData): DiffResult {
    const incomingIds = new Set(gd.nodes.map((n) => n.id));
    const addedNodes: ContextNode[] = [];
    const removedIds: string[] = [];
    const updatedNodes: ContextNode[] = [];

    for (const cn of gd.nodes) {
      if (!this.nodeRegistry.has(cn.id)) {
        addedNodes.push(cn);
      } else {
        const wn = this.nodeRegistry.get(cn.id)!;
        if (
          wn.eventCount !== cn.eventCount ||
          wn.hubDegree !== cn.hubDegree ||
          wn.degree !== cn.degree ||
          wn.kind !== cn.kind
        )
          updatedNodes.push(cn);
      }
    }
    for (const id of this.nodeRegistry.keys())
      if (!incomingIds.has(id)) removedIds.push(id);

    // Early-exit edge diff: bail out on first mismatch rather than building
    // two full Sets and diffing them.
    let edgesChanged =
      gd.allEdges.length !== this.edgeRegistry.size;
    if (!edgesChanged) {
      for (const edge of gd.allEdges) {
        if (!this.edgeRegistry.has(edgeKey(edge))) {
          edgesChanged = true;
          break;
        }
      }
    }

    const onlyVisuals =
      addedNodes.length === 0 && removedIds.length === 0 && !edgesChanged;
    return { addedNodes, removedIds, updatedNodes, edgesChanged, onlyVisuals };
  }

  /** Build a Map<nodeId, neighbourIds[]> from the edge list — O(E) once. */
  private buildAdjacencyIndex(edges: AnyEdge[]): Map<string, string[]> {
    const adj = new Map<string, string[]>();
    for (const edge of edges) {
      let src = adj.get(edge.sourceId);
      if (!src) { src = []; adj.set(edge.sourceId, src); }
      src.push(edge.targetId);

      let tgt = adj.get(edge.targetId);
      if (!tgt) { tgt = []; adj.set(edge.targetId, tgt); }
      tgt.push(edge.sourceId);
    }
    return adj;
  }

  private initialPosition(
    cn: ContextNode,
    gd: ContextGraphData,
    adj: Map<string, string[]>
  ): [number, number] {
    if (gd.nodes.length === 1) return [SPACE / 2, SPACE / 2];

    const neighbours = adj.get(cn.id) ?? [];
    const knownPositions: Array<[number, number]> = [];
    for (const nbId of neighbours) {
      const wn = this.nodeRegistry.get(nbId);
      if (wn) knownPositions.push([wn.cachedX, wn.cachedY]);
    }

    const jitter = () => (Math.random() - 0.5) * SPACE * 0.04;
    if (knownPositions.length > 0) {
      const cx =
        knownPositions.reduce((s, p) => s + p[0], 0) / knownPositions.length;
      const cy =
        knownPositions.reduce((s, p) => s + p[1], 0) / knownPositions.length;
      return [cx + jitter(), cy + jitter()];
    }
    const angle = Math.random() * Math.PI * 2;
    const r = Math.random() * SPACE * 0.08;
    return [SPACE / 2 + Math.cos(angle) * r, SPACE / 2 + Math.sin(angle) * r];
  }

  private flushToCosmosArrays(edges: AnyEdge[]) {
    const slots = this.totalSlots;
    const edgeCount = edges.length;

    // Grow backing arrays only when capacity is exceeded — avoid per-call alloc.
    if (this._pointPositions.length < slots * 2) {
      this._pointPositions = new Float32Array(slots * 2);
      this._pointColors = new Float32Array(slots * 4);
      this._pointSizes = new Float32Array(slots);
    }
    if (this._links.length < edgeCount * 2) {
      this._links = new Float32Array(edgeCount * 2);
      this._linkColors = new Float32Array(edgeCount * 4);
      this._linkWidths = new Float32Array(edgeCount);
    }

    const pointPositions = this._pointPositions;
    const pointColors = this._pointColors;
    const pointSizes = this._pointSizes;

    for (const wn of this.nodeRegistry.values()) {
      const i = wn.cosmosIndex;
      pointPositions[i * 2] = wn.cachedX;
      pointPositions[i * 2 + 1] = wn.cachedY;
      let rgba: [number, number, number, number];
      let size: number;
      if (wn.kind === "hub") {
        rgba = this.hubColorScale(wn.hubDegree);
        size = this.hubSizeScale(wn.eventCount);
      } else if (wn.kind === "event") {
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

    const links = this._links;
    const linkColors = this._linkColors;
    const linkWidths = this._linkWidths;

    for (let i = 0; i < edgeCount; i++) {
      const edge = edges[i];
      links[i * 2] = this.nodeRegistry.get(edge.sourceId)?.cosmosIndex ?? 0;
      links[i * 2 + 1] = this.nodeRegistry.get(edge.targetId)?.cosmosIndex ?? 0;
      if (edge.kind === "hub-hub") {
        const a = this.hhAlphaScale(edge.weight);
        const w = this.hhWidthScale(edge.weight);
        linkColors[i * 4] = HH_LINK_BASE_RGBA[0];
        linkColors[i * 4 + 1] = HH_LINK_BASE_RGBA[1];
        linkColors[i * 4 + 2] = HH_LINK_BASE_RGBA[2];
        linkColors[i * 4 + 3] = a;
        linkWidths[i] = w;
      } else {
        const a = this.spokeAlpha(edge.weight);
        const w = this.spokeWidth(edge.weight);
        linkColors[i * 4] = SPOKE_LINK_RGBA[0];
        linkColors[i * 4 + 1] = SPOKE_LINK_RGBA[1];
        linkColors[i * 4 + 2] = SPOKE_LINK_RGBA[2];
        linkColors[i * 4 + 3] = a;
        linkWidths[i] = w;
      }
    }

    // Pass subarray views so cosmos sees exactly the right length even when
    // the backing buffer is larger than the current node/edge count.
    this.cosmos.setPointPositions(
      slots === this._pointPositions.length / 2
        ? pointPositions
        : pointPositions.subarray(0, slots * 2)
    );
    this.cosmos.setPointColors(
      slots === this._pointColors.length / 4
        ? pointColors
        : pointColors.subarray(0, slots * 4)
    );
    this.cosmos.setPointSizes(
      slots === this._pointSizes.length
        ? pointSizes
        : pointSizes.subarray(0, slots)
    );
    this.cosmos.setLinks(
      edgeCount === this._links.length / 2
        ? links
        : links.subarray(0, edgeCount * 2)
    );
    this.cosmos.setLinkColors(
      edgeCount === this._linkColors.length / 4
        ? linkColors
        : linkColors.subarray(0, edgeCount * 4)
    );
    this.cosmos.setLinkWidths(
      edgeCount === this._linkWidths.length
        ? linkWidths
        : linkWidths.subarray(0, edgeCount)
    );
    this.cosmos.render();
  }
}

// ─── Component ─────────────────────────────────────────────────────────────────

const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

const CosmosComponent: Component = () => {
  const [labelPositions, setLabelPositions] = createSignal<
    Array<{ id: string; label: string; kind: string; x: number; y: number }>
  >([]);
  const [hoveredId, setHoveredId] = createSignal<string | null>(null);

  // ── Year bounds ──────────────────────────────────────────────────────────

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = tier2Data[controls.concept];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });

  // Guard year-bounds effect against cascading reactive updates: only write
  // to controls when the values actually change, and untrack the reads of the
  // current control values to avoid a reactive loop.
  createEffect(() => {
    const [min, max] = yearBounds();
    if (controls.yearMode === "single") {
      const mid = Math.floor((min + max) / 2);
      untrack(() => {
        if (controls.fromYear !== mid) setControls("fromYear", mid);
        if (controls.toYear !== mid) setControls("toYear", mid);
      });
    } else {
      untrack(() => {
        if (controls.fromYear !== min) setControls("fromYear", min);
        if (controls.toYear !== max) setControls("toYear", max);
      });
    }
  });

  const yearFiltered = createMemo(() => {
    const cd = tier2Data[controls.concept];
    if (!cd) return [];
    const [min, max] = yearBounds();
    return controls.fromYear <= min && controls.toYear >= max
      ? cd.events
      : filterByYearRange(cd.events, controls.fromYear, controls.toYear);
  });

  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(yearFiltered())
  );

  const graphData = createMemo<ContextGraphData>(() =>
    controls.viewMode === "events"
      ? buildPureEventGraph(yearFiltered(), controls.topN)
      : buildContextualGraph(
        tokenBins(),
        controls.topN,
        controls.minSimilarity,
        controls.maxHubs
      )
  );

  // ── Drill-down memos ─────────────────────────────────────────────────────

  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
    const id = controls.selectedNode;
    if (!id) return null;
    return graphData().nodes.find((n) => n.id === id)?.kind ?? null;
  });

  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "hub") return null;
    return tokenBins().get(id) ?? null;
  });

  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const bin = selectedBin();
    if (!bin) return [];
    return [...bin.docs.entries()].sort(
      (a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity)
    );
  });

  const selectedEventNode = createMemo<ContextNode | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find((n) => n.id === id) ?? null;
  });

  const sharedByHubs = createMemo<
    Array<{ hub: string; freq: number; meanScore: number }>
  >(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "neighbour") return [];
    if (controls.viewMode === "aggregated") {
      const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
      for (const [hubKey, bin] of tokenBins()) {
        const nb = bin.topNeighbours.find((n) => n.token === id);
        if (nb) result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore });
      }
      return result.sort((a, b) => b.freq - a.freq);
    }
    return graphData()
      .hubNbEdges.filter((e) => e.targetId === id)
      .map((e) => ({ hub: e.sourceId, freq: e.weight, meanScore: e.weight }))
      .sort((a, b) => b.freq - a.freq);
  });

  // ── Cosmos + GraphWorld refs ─────────────────────────────────────────────

  let wrapRef!: HTMLDivElement;
  let cosmosGraph: Graph | null = null;
  let world: GraphWorld | null = null;
  let rafHandle = 0;

  let mouseClientX = 0;
  let mouseClientY = 0;

  // ── Tooltip ──────────────────────────────────────────────────────────────

  let tooltipEl: HTMLDivElement | null = null;
  const getTooltip = (): HTMLDivElement => {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className =
        "cg-tooltip surface-container-high border large-elevate padding";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  };

  // Pure imperative tooltip — does NOT call reactive memos so hovering
  // never triggers SolidJS batch updates.
  function showTooltip(wn: WorldNode) {
    const tip = getTooltip();
    let html = "";
    if (wn.kind === "hub") {
      const bin = untrack(tokenBins).get(wn.id);
      const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
      const yStr = years.length
        ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }`
        : "—";
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Events: ${ wn.eventCount }<br/>Connections: ${ wn.hubDegree }<br/>Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`;
    } else if (wn.kind === "event") {
      html = `<aside><h6 class="bottom-padding">${ wn.token ?? wn.id }</h6>Doc: ${ wn.doc_id ?? "—" }<br/>Year: ${ wn.pub_year ?? "—" }</aside>`;
    } else {
      // For neighbour tooltips we need a snapshot of sharedByHubs without
      // subscribing to its reactive dependency chain.  We compute it directly
      // from the current untracked graph state instead of calling the memo.
      const id = wn.id;
      const viewMode = untrack(() => controls.viewMode);
      const bins = untrack(tokenBins);
      const gd = untrack(graphData);
      let hubs: Array<{ hub: string; freq: number }>;
      if (viewMode === "aggregated") {
        hubs = [];
        for (const [hubKey, bin] of bins) {
          const nb = bin.topNeighbours.find((n) => n.token === id);
          if (nb) hubs.push({ hub: hubKey, freq: nb.freq });
        }
        hubs.sort((a, b) => b.freq - a.freq);
      } else {
        hubs = gd.hubNbEdges
          .filter((e) => e.targetId === id)
          .map((e) => ({ hub: e.sourceId, freq: e.weight }))
          .sort((a, b) => b.freq - a.freq);
      }
      const lines = hubs.length
        ? hubs
          .slice(0, 5)
          .map((h) => `${ h.hub } (${ h.freq.toFixed(3) })`)
          .join("<br/>")
        : "—";
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Shared by ${ wn.degree } source(s):<br/>${ lines }</aside>`;
    }
    tip.innerHTML = html;
    Object.assign(tip.style, {
      opacity: "1",
      left: `${ mouseClientX + 14 }px`,
      top: `${ mouseClientY - 10 }px`,
    });
  }

  // ── Label rAF loop ────────────────────────────────────────────────────────
  //
  // Changed from original:
  //  1. We keep a stable cache of the previous frame's label positions keyed
  //     by cosmosIndex.  If no node moved more than LABEL_MOVE_THRESHOLD px
  //     and the node set is the same size, we skip setLabelPositions entirely.
  //  2. The "concept hub suppression" logic reads concept via untrack so the
  //     RAF closure does NOT subscribe to the controls store (would re-run
  //     the entire effect on every keypress in a search box, etc.).

  const prevLabelCache = new Map<number, { x: number; y: number }>();

  function startLabelLoop() {
    cancelAnimationFrame(rafHandle);

    function tick() {
      if (!cosmosGraph || !world) {
        rafHandle = requestAnimationFrame(tick);
        return;
      }
      const posMap = cosmosGraph.getTrackedPointPositionsMap();

      if (posMap && posMap.size > 0) {
        const hid = hoveredId();
        const currentConcept = untrack(() => controls.concept);
        const showEventLabels = untrack(() => controls.showEventLabels);

        let dirty = posMap.size !== prevLabelCache.size;
        const nextLabels: Array<{ id: string; label: string; kind: string; x: number; y: number }> =
          [];

        posMap.forEach(([sx, sy], idx) => {
          const wn = world!.getNodeByIndex(idx);
          if (!wn) return;
          const isConceptHub = wn.kind === "hub" && wn.id === currentConcept;
          if (isConceptHub && wn.id !== hid) return;
          // Skip event labels unless the toggle is on, or this node is hovered.
          if (wn.kind === "event" && !showEventLabels && wn.id !== hid) return;

          // Convert simulation-space coords → canvas-element px.
          const [x, y] = cosmosGraph!.spaceToScreenPosition([sx, sy]);

          // For event nodes show the human-readable token, not the synthetic id.
          const label = wn.kind === "event" ? (wn.token ?? wn.id) : wn.id;

          nextLabels.push({ id: wn.id, label, kind: wn.kind, x, y });

          if (!dirty) {
            const prev = prevLabelCache.get(idx);
            if (
              !prev ||
              Math.abs(prev.x - x) > LABEL_MOVE_THRESHOLD ||
              Math.abs(prev.y - y) > LABEL_MOVE_THRESHOLD
            ) {
              dirty = true;
            }
          }
        });

        if (dirty) {
          prevLabelCache.clear();
          for (const lbl of nextLabels) {
            // We need the cosmosIndex here; look it up once via the world.
            const wn = world!.getNodeById(lbl.id);
            if (wn) prevLabelCache.set(wn.cosmosIndex, { x: lbl.x, y: lbl.y });
          }
          setLabelPositions(nextLabels);
        }
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

      onPointMouseOver: (index: number) => {
        const wn = world?.getNodeByIndex(index);
        if (!wn) return;
        setHoveredId(wn.id);
        showTooltip(wn);
      },
      onPointMouseOut: () => {
        setHoveredId(null);
        if (tooltipEl) tooltipEl.style.opacity = "0";
      },
      onClick: (index: number | undefined) => {
        if (index === undefined) {
          setControls("selectedNode", null);
          return;
        }
        const wn = world?.getNodeByIndex(index);
        if (!wn) return;
        setControls("selectedNode", (prev) => (prev === wn.id ? null : wn.id));
      },
    });

    world = new GraphWorld(cosmosGraph);

    wrapRef.addEventListener("mousemove", (e: MouseEvent) => {
      mouseClientX = e.clientX;
      mouseClientY = e.clientY;
    });

    startLabelLoop();
  }

  // ── Effect 1: diff on graphData / concept change ─────────────────────────

  let lastConcept = "";

  createEffect(() => {
    const gd = graphData();
    const currentConcept = controls.concept;

    if (!wrapRef) return;
    ensureCosmosInitialised();

    if (currentConcept !== lastConcept) {
      world!.reset();
      lastConcept = currentConcept;
    }

    if (gd.nodes.length === 0) {
      world!.reset();
      setLabelPositions([]);
      return;
    }

    const hubCount = gd.nodes.filter(
      (n) => n.kind === "hub" || n.kind === "event"
    ).length;
    const isDegenerate =
      gd.allEdges.length === 0 || gd.nodes.length <= 1 || hubCount <= 1;

    // Read hubSpread without subscribing — the hubSpread effect owns that
    // dependency so this effect doesn't fire twice when spread changes.
    const { hubSpread } = untrack(() => controls);

    cosmosGraph!.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: isDegenerate
        ? 0
        : BASE_REPULSION * (0.6 + hubSpread * 0.4),
      simulationLinkSpring: isDegenerate ? 0 : BASE_LINK_SPRING,
      simulationGravity: isDegenerate ? 0.8 : BASE_GRAVITY,
      simulationFriction: BASE_FRICTION,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white",
      renderHoveredPointRing: true,
    });

    const topologyChanged = world!.applyDiff(gd);
    if (topologyChanged) {
      cosmosGraph!.start();
      setTimeout(() => cosmosGraph?.fitView(), 300);
    }
  });

  // ── Effect 2: hubSpread force tweak ──────────────────────────────────────
  // Reads graphData() via untrack so topology changes don't re-run this
  // effect — only hubSpread changes should.

  createEffect(() => {
    const hubSpread = controls.hubSpread;
    if (!cosmosGraph || !world) return;

    const gd = untrack(graphData);

    const hubCount = gd.nodes.filter(
      (n) => n.kind === "hub" || n.kind === "event"
    ).length;
    if (gd.allEdges.length === 0 || gd.nodes.length <= 1 || hubCount <= 1)
      return;

    cosmosGraph.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: BASE_REPULSION * hubSpread,
      simulationLinkSpring: Math.min(0.92, BASE_FRICTION + hubSpread * 0.05),
      simulationFriction: BASE_FRICTION,
      simulationGravity: BASE_GRAVITY,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white",
      renderHoveredPointRing: true,
    });
  });

  onCleanup(() => {
    cancelAnimationFrame(rafHandle);
    cosmosGraph?.pause();
    tooltipEl?.remove();
    tooltipEl = null;
  });

  const totalEventsForConcept = createMemo(() => {
    const cd = tier2Data[controls.concept];
    return cd?.n_events ?? 0;
  });

  return (
    <>
      <div class="cg-layout">
        <ControlsHeader
          totalEvents={totalEventsForConcept}
          includeHubSpread={true}
        />

        {/* ── Main canvas + aside ── */}
        <div class="cg-main background">
          <div ref={wrapRef!} class="cg-canvas-wrap surface-container">
            {/* Label overlay */}
            <div class="cg-labels">
              <For each={labelPositions()}>
                {(lbl) => (
                  <span
                    class={`cg-label ${ lbl.kind }`}
                    style={{
                      left: `${ lbl.x }px`,
                      top: `${ lbl.y + 14 }px`,
                    }}
                  >
                    {lbl.label}
                  </span>
                )}
              </For>
            </div>

            {/* Empty-state */}
            <Show when={graphData().nodes.length === 0}>
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  "align-items": "center",
                  "justify-content": "center",
                  "pointer-events": "none",
                }}
              >
                <span class="error">
                  No graph: try reducing min similarity or increasing top N
                </span>
              </div>
            </Show>
          </div>

          {/* ── Drill-down aside ── */}
          <Show when={controls.selectedNode}>
            <aside class="cg-aside surface-container-high padding border">
              <div class="cg-header-row">
                <h2>{controls.selectedNode}</h2>
                <button
                  class="link border"
                  onClick={() => setControls("selectedNode", null)}
                >
                  ✕
                </button>
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
                        <div>
                          Years:{" "}
                          {years.length
                            ? years.length === 1
                              ? years[0]
                              : `${ years[0] }–${ years[years.length - 1] }`
                            : "—"}
                        </div>
                        <div>
                          Hub connections:{" "}
                          {graphData().nodes.find(
                            (n) => n.id === controls.selectedNode
                          )?.hubDegree ?? 0}
                        </div>
                      </div>
                      <h3 class="bottom-padding">Top neighbours</h3>
                      <div class="bottom-padding">
                        <For each={bin.topNeighbours.slice(0, MAX_TOP_N)}>
                          {(nb) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div
                                  class="cg-nb-bar-fill hub"
                                  style={{
                                    width: `${ (nb.freq / topMax) * 100 }%`,
                                  }}
                                />
                              </div>
                              <span class="cg-nb-token">{nb.token}</span>
                              <span class="cg-nb-score">
                                {nb.meanScore.toFixed(3)}
                              </span>
                            </div>
                          )}
                        </For>
                      </div>
                      <h3 class="bottom-padding">Sources</h3>
                      <Show
                        when={selectedDocs().length > 0}
                        fallback={
                          <div class="error">No documents found</div>
                        }
                      >
                        <For each={selectedDocs()}>
                          {([docId, pubYear]) => (
                            <button
                              class="chip small-margin cg-chip-mono"
                              onClick={() => showDocument(docId)}
                            >
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
                            <button
                              class="chip small-margin cg-chip-mono"
                              onClick={() => showDocument(node.doc_id!)}
                            >
                              <span>{node.doc_id}</span>
                            </button>
                          </div>
                        </Show>
                      </div>
                      <div
                        class="bottom-padding small-text"
                        style={{ opacity: 0.6 }}
                      >
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
                  {controls.viewMode === "aggregated"
                    ? "Hub contexts"
                    : "Event contexts"}
                </h3>
                <Show
                  when={sharedByHubs().length > 0}
                  fallback={
                    <div class="error">Not in any top-N list</div>
                  }
                >
                  {(_) => {
                    const maxFreq = sharedByHubs()[0]?.freq ?? 1;
                    return (
                      <div class="bottom-padding">
                        <For each={sharedByHubs()}>
                          {(h) => (
                            <div class="cg-nb-row">
                              <div class="cg-nb-bar-wrap">
                                <div
                                  class="cg-nb-bar-fill neighbour"
                                  style={{
                                    width: `${ (h.freq / maxFreq) * 100 }%`,
                                  }}
                                />
                              </div>
                              <span class="cg-nb-token">{h.hub}</span>
                              <span class="cg-nb-score">
                                {h.meanScore.toFixed(3)}
                              </span>
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
            <Show when={controls.viewMode === "aggregated"}>
              <span class="cg-legend-hub" />
              hubs ({graphData().nodes.filter((n) => n.kind === "hub").length})
            </Show>
            <Show when={controls.viewMode === "events"}>
              <span class="cg-legend-event" />
              events (
              {graphData().nodes.filter((n) => n.kind === "event").length})
            </Show>
            <span class="cg-legend-nb" />
            neighbours (
            {graphData().nodes.filter((n) => n.kind === "neighbour").length})
            <Show when={controls.viewMode === "aggregated"}>
              {" • "}
              {graphData().hubHubEdges.length} similarity edges
            </Show>
            {" • "}
            {graphData().hubNbEdges.length} spokes{" • "}
            {yearFiltered().length} events
            <Show
              when={
                controls.fromYear !== yearBounds()[0] ||
                controls.toYear !== yearBounds()[1]
              }
            >
              {" • "}
              {controls.fromYear}–{controls.toYear}
            </Show>
          </span>
        </footer>
      </div>
    </>
  );
};

export default CosmosComponent;