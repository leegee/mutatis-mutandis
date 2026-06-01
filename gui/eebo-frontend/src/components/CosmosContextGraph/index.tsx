import {
  createSignal,
  createMemo,
  createResource,
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

import type {
  AnyEdge, ContextGraphData, ContextNode, TokenBin,
} from "./types";

import { controls, setControls } from "../../state/controls.store";
import { aggregateByToken, buildContextualGraph, buildPureEventGraph, } from "../../lib/contextGraphUtils";
import { getYearFiltered, getYearBounds, totalEventsForConcept } from "../../state/selectors";
import { showDocument } from "../../services/documentApi";
import ControlsHeader from "../ControlsHeader";
import EventContext from "../EventContext";

const MAX_TOP_N = 20;

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

const LABEL_MOVE_THRESHOLD = 0.5;

const HUB_COLOR_LOW_RGBA: [number, number, number, number] = [0.35, 0.53, 0.73, 0.4];
const HUB_COLOR_HIGH_RGBA: [number, number, number, number] = [0.91, 0.95, 0.99, 0.87];
const EVENT_RGBA: [number, number, number, number] = [0.47, 0.82, 0.51, 0.75];
const NB_RGBA: [number, number, number, number] = [1.0, 0.75, 0.31, 0.65];
const HH_LINK_BASE_RGBA: [number, number, number, number] = [0.55, 0.75, 0.95, 0.95];
const SPOKE_LINK_RGBA: [number, number, number, number] = [1.0, 0.75, 0.31, 0.9];

const HH_WIDTH_MIN = 2; const HH_WIDTH_MAX = 3;
const HH_ALPHA_MIN = 0.75; const HH_ALPHA_MAX = 1;
const SPOKE_ALPHA_MIN = 0.75; const SPOKE_ALPHA_MAX = 0.99;
const SPOKE_WIDTH_MIN = 2; const SPOKE_WIDTH_MAX = 3;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [], hubHubEdges: [], hubNbEdges: [], allEdges: [],
  maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1,
};

// GraphWorld — unchanged from original
interface WorldNode extends ContextNode { cosmosIndex: number; cachedX: number; cachedY: number; }
interface WorldEdge { key: string; edge: AnyEdge; cosmosRow: number; }
type DiffResult = { addedNodes: ContextNode[]; removedIds: string[]; updatedNodes: ContextNode[]; edgesChanged: boolean; onlyVisuals: boolean; };

function edgeKey(e: AnyEdge): string { return `${ e.sourceId }→${ e.targetId }`; }

class GraphWorld {
  private nodeRegistry = new Map<string, WorldNode>();
  private reverseIndex = new Map<number, WorldNode>();
  private edgeRegistry = new Map<string, WorldEdge>();
  private indexPool: number[] = [];
  private nextIndex = 0;
  private totalSlots = 0;
  private cosmos: Graph;
  private _pointPositions = new Float32Array(0);
  private _pointColors = new Float32Array(0);
  private _pointSizes = new Float32Array(0);
  private _links = new Float32Array(0);
  private _linkColors = new Float32Array(0);
  private _linkWidths = new Float32Array(0);
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
    const adjIndex = diff.addedNodes.length > 0 ? this.buildAdjacencyIndex(gd.allEdges) : null;
    for (const cn of diff.addedNodes) {
      const idx = this.indexPool.length > 0 ? this.indexPool.pop()! : this.nextIndex++;
      this.totalSlots = Math.max(this.totalSlots, idx + 1);
      const [ix, iy] = this.initialPosition(cn, gd, adjIndex!);
      const wn: WorldNode = { ...cn, cosmosIndex: idx, cachedX: ix, cachedY: iy };
      this.nodeRegistry.set(cn.id, wn);
      this.reverseIndex.set(idx, wn);
    }
    for (const cn of diff.updatedNodes) Object.assign(this.nodeRegistry.get(cn.id)!, cn);
    if (diff.edgesChanged) {
      this.edgeRegistry.clear();
      gd.allEdges.forEach((edge, row) => this.edgeRegistry.set(edgeKey(edge), { key: edgeKey(edge), edge, cosmosRow: row }));
    }
    this.flushToCosmosArrays(gd.allEdges);
    this.cosmos.trackPointPositionsByIndices([...this.nodeRegistry.values()].map((wn) => wn.cosmosIndex));
    return !diff.onlyVisuals;
  }

  reset() {
    this.nodeRegistry.clear(); this.reverseIndex.clear(); this.edgeRegistry.clear();
    this.indexPool = []; this.nextIndex = 0; this.totalSlots = 0;
    this.cosmos.setPointPositions(new Float32Array(0));
    this.cosmos.setLinks(new Float32Array(0));
    this.cosmos.render();
  }

  snapshotPositions() {
    const posMap = this.cosmos.getTrackedPointPositionsMap();
    if (!posMap) return;
    posMap.forEach(([x, y], idx) => { const wn = this.reverseIndex.get(idx); if (wn) { wn.cachedX = x; wn.cachedY = y; } });
  }

  getNodeByIndex(idx: number): WorldNode | undefined { return this.reverseIndex.get(idx); }
  getNodeById(id: string): WorldNode | undefined { return this.nodeRegistry.get(id); }
  get liveNodes(): IterableIterator<WorldNode> { return this.nodeRegistry.values(); }

  private rebuildScales(gd: ContextGraphData) {
    this.hubColorScale = d3.scaleLinear<[number, number, number, number]>().domain([0, Math.max(1, gd.maxHubDegree)]).range([HUB_COLOR_LOW_RGBA, HUB_COLOR_HIGH_RGBA]);
    this.hubSizeScale = d3.scaleSqrt().domain([0, gd.maxEventCount]).range([HUB_MIN_SIZE, HUB_MAX_SIZE]);
    this.hhAlphaScale = d3.scaleLinear().domain([0, gd.maxHubHubWeight]).range([HH_ALPHA_MIN, HH_ALPHA_MAX]);
    this.hhWidthScale = d3.scaleLinear().domain([0, gd.maxHubHubWeight]).range([HH_WIDTH_MIN, HH_WIDTH_MAX]);
    this.spokeAlpha = d3.scaleLinear().domain([0, 1]).range([SPOKE_ALPHA_MIN, SPOKE_ALPHA_MAX]);
    this.spokeWidth = d3.scaleLinear().domain([0, 1]).range([SPOKE_WIDTH_MIN, SPOKE_WIDTH_MAX]);
  }

  private computeDiff(gd: ContextGraphData): DiffResult {
    const incomingIds = new Set(gd.nodes.map((n) => n.id));
    const addedNodes: ContextNode[] = []; const removedIds: string[] = []; const updatedNodes: ContextNode[] = [];
    for (const cn of gd.nodes) {
      if (!this.nodeRegistry.has(cn.id)) { addedNodes.push(cn); }
      else { const wn = this.nodeRegistry.get(cn.id)!; if (wn.eventCount !== cn.eventCount || wn.hubDegree !== cn.hubDegree || wn.degree !== cn.degree || wn.kind !== cn.kind) updatedNodes.push(cn); }
    }
    for (const id of this.nodeRegistry.keys()) if (!incomingIds.has(id)) removedIds.push(id);
    let edgesChanged = gd.allEdges.length !== this.edgeRegistry.size;
    if (!edgesChanged) for (const edge of gd.allEdges) if (!this.edgeRegistry.has(edgeKey(edge))) { edgesChanged = true; break; }
    return { addedNodes, removedIds, updatedNodes, edgesChanged, onlyVisuals: addedNodes.length === 0 && removedIds.length === 0 && !edgesChanged };
  }

  private buildAdjacencyIndex(edges: AnyEdge[]): Map<string, string[]> {
    const adj = new Map<string, string[]>();
    for (const edge of edges) {
      let src = adj.get(edge.sourceId); if (!src) { src = []; adj.set(edge.sourceId, src); } src.push(edge.targetId);
      let tgt = adj.get(edge.targetId); if (!tgt) { tgt = []; adj.set(edge.targetId, tgt); } tgt.push(edge.sourceId);
    }
    return adj;
  }

  private initialPosition(cn: ContextNode, gd: ContextGraphData, adj: Map<string, string[]>): [number, number] {
    if (gd.nodes.length === 1) return [SPACE / 2, SPACE / 2];
    const neighbours = adj.get(cn.id) ?? [];
    const known: Array<[number, number]> = [];
    for (const nbId of neighbours) { const wn = this.nodeRegistry.get(nbId); if (wn) known.push([wn.cachedX, wn.cachedY]); }
    const jitter = () => (Math.random() - 0.5) * SPACE * 0.04;
    if (known.length > 0) { const cx = known.reduce((s, p) => s + p[0], 0) / known.length; const cy = known.reduce((s, p) => s + p[1], 0) / known.length; return [cx + jitter(), cy + jitter()]; }
    const angle = Math.random() * Math.PI * 2; const r = Math.random() * SPACE * 0.08;
    return [SPACE / 2 + Math.cos(angle) * r, SPACE / 2 + Math.sin(angle) * r];
  }

  private flushToCosmosArrays(edges: AnyEdge[]) {
    const slots = this.totalSlots; const edgeCount = edges.length;
    if (this._pointPositions.length < slots * 2) { this._pointPositions = new Float32Array(slots * 2); this._pointColors = new Float32Array(slots * 4); this._pointSizes = new Float32Array(slots); }
    if (this._links.length < edgeCount * 2) { this._links = new Float32Array(edgeCount * 2); this._linkColors = new Float32Array(edgeCount * 4); this._linkWidths = new Float32Array(edgeCount); }
    const pp = this._pointPositions; const pc = this._pointColors; const ps = this._pointSizes;
    for (const wn of this.nodeRegistry.values()) {
      const i = wn.cosmosIndex; pp[i * 2] = wn.cachedX; pp[i * 2 + 1] = wn.cachedY;
      let rgba: [number, number, number, number]; let size: number;
      if (wn.kind === "hub") { rgba = this.hubColorScale(wn.hubDegree); size = this.hubSizeScale(wn.eventCount); }
      else if (wn.kind === "event") { rgba = EVENT_RGBA; size = EVENT_SIZE; }
      else { rgba = NB_RGBA; size = NB_POINT_SIZE; }
      pc[i * 4] = rgba[0]; pc[i * 4 + 1] = rgba[1]; pc[i * 4 + 2] = rgba[2]; pc[i * 4 + 3] = rgba[3]; ps[i] = size;
    }
    const lk = this._links; const lc = this._linkColors; const lw = this._linkWidths;
    for (let i = 0; i < edgeCount; i++) {
      const edge = edges[i];
      lk[i * 2] = this.nodeRegistry.get(edge.sourceId)?.cosmosIndex ?? 0;
      lk[i * 2 + 1] = this.nodeRegistry.get(edge.targetId)?.cosmosIndex ?? 0;
      if (edge.kind === "hub-hub") {
        lc[i * 4] = HH_LINK_BASE_RGBA[0]; lc[i * 4 + 1] = HH_LINK_BASE_RGBA[1]; lc[i * 4 + 2] = HH_LINK_BASE_RGBA[2]; lc[i * 4 + 3] = this.hhAlphaScale(edge.weight); lw[i] = this.hhWidthScale(edge.weight);
      } else {
        lc[i * 4] = SPOKE_LINK_RGBA[0]; lc[i * 4 + 1] = SPOKE_LINK_RGBA[1]; lc[i * 4 + 2] = SPOKE_LINK_RGBA[2]; lc[i * 4 + 3] = this.spokeAlpha(edge.weight); lw[i] = this.spokeWidth(edge.weight);
      }
    }
    const sub = <T extends Float32Array>(arr: T, count: number, stride: number): T => count * stride === arr.length ? arr : arr.subarray(0, count * stride) as T;
    this.cosmos.setPointPositions(sub(pp, slots, 2)); this.cosmos.setPointColors(sub(pc, slots, 4)); this.cosmos.setPointSizes(sub(ps, slots, 1));
    this.cosmos.setLinks(sub(lk, edgeCount, 2)); this.cosmos.setLinkColors(sub(lc, edgeCount, 4)); this.cosmos.setLinkWidths(sub(lw, edgeCount, 1));
    this.cosmos.render();
  }
}

// -----------
// Component
// -----------

const CosmosComponent: Component = () => {
  const [labelPositions, setLabelPositions] = createSignal<Array<{ id: string; label: string; kind: string; x: number; y: number }>>([]);
  const [hoveredId, setHoveredId] = createSignal<string | null>(null);
  const [simulating, setSimulating] = createSignal(false);
  let simTimeoutHandle = 0;
  let simGeneration = 0;

  function stopSimulating(generation: number) {
    if (generation !== simGeneration) return;
    clearTimeout(simTimeoutHandle); setSimulating(false);
    cosmosGraph?.pause(); cosmosGraph?.fitView();
  }
  function startSimulating(nodeCount = 0) {
    clearTimeout(simTimeoutHandle);
    const generation = ++simGeneration; setSimulating(true);
    const ms = Math.min(12_000, Math.max(1_500, nodeCount * 10));
    simTimeoutHandle = window.setTimeout(() => stopSimulating(generation), ms);
  }

  // Resources -------------------------------------------------------------

  const [filteredEventsResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredEvents = () => filteredEventsResource() ?? [];

  const [yearBoundsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBounds(concept),
  );
  const yearBounds = (): [number, number] => yearBoundsResource() ?? [controls.fromYear, controls.toYear];

  const [totalEventsResource] = createResource(
    () => controls.concept,
    (concept) => totalEventsForConcept(concept),
  );
  const totalEvents = () => totalEventsResource() ?? 0;

  // Derived memos ---------------------------------------------------------

  const tokenBins = createMemo<Map<string, TokenBin>>(() => aggregateByToken(filteredEvents()));

  const graphData = createMemo<ContextGraphData>(() =>
    controls.viewMode === "events"
      ? buildPureEventGraph(filteredEvents(), controls.topN, EMPTY_GRAPH)
      : buildContextualGraph(tokenBins(), controls.topN, controls.minSimilarity, controls.maxHubs, EMPTY_GRAPH)
  );

  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
    const id = controls.selectedNode; if (!id) return null;
    return graphData().nodes.find((n) => n.id === id)?.kind ?? null;
  });

  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "hub") return null;
    return tokenBins().get(id) ?? null;
  });

  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const bin = selectedBin(); if (!bin) return [];
    return [...bin.docs.entries()].sort((a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity));
  });

  const selectedEventNode = createMemo<ContextNode | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find((n) => n.id === id) ?? null;
  });

  const sharedByHubs = createMemo<Array<{ hub: string; freq: number; meanScore: number }>>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "neighbour") return [];
    if (controls.viewMode === "aggregated") {
      const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
      for (const [hubKey, bin] of tokenBins()) { const nb = bin.topNeighbours.find((n) => n.token === id); if (nb) result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore }); }
      return result.sort((a, b) => b.freq - a.freq);
    }
    return graphData().hubNbEdges.filter((e) => e.targetId === id).map((e) => ({ hub: e.sourceId, freq: e.weight, meanScore: e.weight })).sort((a, b) => b.freq - a.freq);
  });

  // Cosmos refs -----------------------------------------------------------

  let wrapRef!: HTMLDivElement;
  let cosmosGraph: Graph | null = null;
  let world: GraphWorld | null = null;
  let rafHandle = 0;
  let mouseClientX = 0;
  let mouseClientY = 0;
  let tooltipEl: HTMLDivElement | null = null;

  const getTooltip = (): HTMLDivElement => {
    if (!tooltipEl) { tooltipEl = document.createElement("div"); tooltipEl.className = "cg-tooltip surface-container-highest border large-elevate padding"; document.body.appendChild(tooltipEl); }
    return tooltipEl;
  };

  function showTooltip(wn: WorldNode) {
    const tip = getTooltip(); let html = "";
    if (wn.kind === "hub") {
      const bin = untrack(tokenBins).get(wn.id); const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
      const yStr = years.length ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }` : "—";
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Events: ${ wn.eventCount }<br/>Connections: ${ wn.hubDegree }<br/>Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`;
    } else if (wn.kind === "event") {
      html = `<aside><h6 class="bottom-padding">${ wn.token ?? wn.id }</h6>Doc: ${ wn.doc_id ?? "—" }<br/>Year: ${ wn.pub_year ?? "—" }<br/>token_idx: ${ wn.token_idx ?? "—" }</aside>`;
    } else {
      const id = wn.id; const viewMode = untrack(() => controls.viewMode); const bins = untrack(tokenBins); const gd = untrack(graphData);
      let hubs: Array<{ hub: string; freq: number }>;
      if (viewMode === "aggregated") { hubs = []; for (const [hubKey, bin] of bins) { const nb = bin.topNeighbours.find((n) => n.token === id); if (nb) hubs.push({ hub: hubKey, freq: nb.freq }); } hubs.sort((a, b) => b.freq - a.freq); }
      else { hubs = gd.hubNbEdges.filter((e) => e.targetId === id).map((e) => ({ hub: e.sourceId, freq: e.weight })).sort((a, b) => b.freq - a.freq); }
      html = `<aside><h6 class="bottom-padding">${ wn.id }</h6>Shared by ${ wn.degree } source(s):<br/>${ hubs.length ? hubs.slice(0, 5).map((h) => `${ h.hub } (${ h.freq.toFixed(3) })`).join("<br/>") : "—" }</aside>`;
    }
    tip.innerHTML = html;
    Object.assign(tip.style, { opacity: "1", left: `${ mouseClientX + 14 }px`, top: `${ mouseClientY - 10 }px` });
  }

  const prevLabelCache = new Map<number, { x: number; y: number }>();

  function startLabelLoop() {
    cancelAnimationFrame(rafHandle);
    function tick() {
      if (!cosmosGraph || !world) { rafHandle = requestAnimationFrame(tick); return; }
      const posMap = cosmosGraph.getTrackedPointPositionsMap();
      if (posMap && posMap.size > 0) {
        const hid = hoveredId(); const currentConcept = untrack(() => controls.concept); const showEventLabels = untrack(() => controls.showEventLabels);
        let dirty = posMap.size !== prevLabelCache.size;
        const nextLabels: Array<{ id: string; label: string; kind: string; x: number; y: number }> = [];
        posMap.forEach(([sx, sy], idx) => {
          const wn = world!.getNodeByIndex(idx); if (!wn) return;
          if (wn.kind === "hub" && wn.id === currentConcept && wn.id !== hid) return;
          if (wn.kind === "event" && !showEventLabels && wn.id !== hid) return;
          const [x, y] = cosmosGraph!.spaceToScreenPosition([sx, sy]);
          const label = wn.kind === "event" ? (wn.token ?? wn.id) : wn.id;
          nextLabels.push({ id: wn.id, label, kind: wn.kind, x, y });
          if (!dirty) { const prev = prevLabelCache.get(idx); if (!prev || Math.abs(prev.x - x) > LABEL_MOVE_THRESHOLD || Math.abs(prev.y - y) > LABEL_MOVE_THRESHOLD) dirty = true; }
        });
        if (dirty) {
          prevLabelCache.clear();
          for (const lbl of nextLabels) { const wn = world!.getNodeById(lbl.id); if (wn) prevLabelCache.set(wn.cosmosIndex, { x: lbl.x, y: lbl.y }); }
          setLabelPositions(nextLabels);
        }
      }
      rafHandle = requestAnimationFrame(tick);
    }
    rafHandle = requestAnimationFrame(tick);
  }

  function ensureCosmosInitialised() {
    if (cosmosGraph) return;
    cosmosGraph = new Graph(wrapRef, {
      backgroundColor: "#0a0c10", spaceSize: SPACE, enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING, simulationRepulsion: BASE_REPULSION,
      simulationLinkSpring: BASE_LINK_SPRING, simulationFriction: BASE_FRICTION,
      simulationGravity: BASE_GRAVITY, pointDefaultSize: NB_POINT_SIZE,
      linkDefaultWidth: 1, hoveredPointRingColor: "white", renderHoveredPointRing: true,
      onSimulationEnd: () => stopSimulating(simGeneration),
      onPointMouseOver: (index: number) => { const wn = world?.getNodeByIndex(index); if (!wn) return; setHoveredId(wn.id); showTooltip(wn); },
      onPointMouseOut: () => { setHoveredId(null); if (tooltipEl) tooltipEl.style.opacity = "0"; },
      onClick: (index: number | undefined) => {
        if (index === undefined) { setControls("selectedNode", null); return; }
        const wn = world?.getNodeByIndex(index); if (!wn) return;
        setControls("selectedNode", (prev) => (prev === wn.id ? null : wn.id));
      },
    });
    world = new GraphWorld(cosmosGraph);
    wrapRef.addEventListener("mousemove", (e: MouseEvent) => { mouseClientX = e.clientX; mouseClientY = e.clientY; });
    startLabelLoop();
  }

  let lastConcept = "";

  createEffect(() => {
    const gd = graphData(); const currentConcept = controls.concept;
    if (!wrapRef) return;
    ensureCosmosInitialised();
    if (currentConcept !== lastConcept) { world!.reset(); lastConcept = currentConcept; }
    if (gd.nodes.length === 0) { world!.reset(); setLabelPositions([]); return; }
    const hubCount = gd.nodes.filter((n) => n.kind === "hub" || n.kind === "event").length;
    const isDegenerate = gd.allEdges.length === 0 || gd.nodes.length <= 1 || hubCount <= 1;
    const { hubSpread } = untrack(() => controls);
    cosmosGraph!.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: isDegenerate ? 0 : BASE_REPULSION * (0.6 + hubSpread * 0.4),
      simulationLinkSpring: isDegenerate ? 0 : BASE_LINK_SPRING,
      simulationGravity: isDegenerate ? 0.8 : BASE_GRAVITY,
      simulationFriction: BASE_FRICTION, enableDrag: true, fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white", renderHoveredPointRing: true,
    });
    const topologyChanged = world!.applyDiff(gd);
    if (topologyChanged) { startSimulating(gd.nodes.length); cosmosGraph!.start(); }
  });

  createEffect(() => {
    const hubSpread = controls.hubSpread;
    if (!cosmosGraph || !world) return;
    const gd = untrack(graphData);
    const hubCount = gd.nodes.filter((n) => n.kind === "hub" || n.kind === "event").length;
    if (gd.allEdges.length === 0 || gd.nodes.length <= 1 || hubCount <= 1) return;
    cosmosGraph.setConfig({
      spaceSize: SPACE, simulationRepulsion: BASE_REPULSION * hubSpread,
      simulationLinkSpring: Math.min(0.92, BASE_FRICTION + hubSpread * 0.05),
      simulationFriction: BASE_FRICTION, simulationGravity: BASE_GRAVITY,
      enableDrag: true, fitViewPadding: FIT_VIEW_PADDING,
      hoveredPointRingColor: "white", renderHoveredPointRing: true,
    });
  });

  onCleanup(() => {
    cancelAnimationFrame(rafHandle); clearTimeout(simTimeoutHandle);
    cosmosGraph?.pause(); tooltipEl?.remove(); tooltipEl = null;
  });


  return (
    <>
      <div class="cg-layout">
        <ControlsHeader totalEvents={totalEvents} includeHubSpread={true} />

        <div class="cg-main background">
          <div ref={wrapRef!} class="cg-canvas-wrap surface-container" style={{ visibility: simulating() ? "hidden" : "visible" }}>
            <div class="cg-labels">
              <For each={labelPositions()}>
                {(lbl) => <span class={`cg-label ${ lbl.kind }`} style={{ left: `${ lbl.x }px`, top: `${ lbl.y + 14 }px` }}>{lbl.label}</span>}
              </For>
            </div>
            <Show when={graphData().nodes.length === 0}>
              <div style={{ position: "absolute", inset: 0, display: "flex", "align-items": "center", "justify-content": "center", "pointer-events": "none" }}>
                <span class="error">No graph: try reducing min similarity or increasing top N</span>
              </div>
            </Show>
          </div>

          <Show when={simulating()}>
            <div style={{ position: "absolute", inset: 0, display: "flex", "flex-direction": "column", "align-items": "center", "justify-content": "center", gap: "1rem", "pointer-events": "none" }}>
              <h3>Settling layout…</h3>
              <progress class="circle light-green-text" />
            </div>
          </Show>

          <Show when={controls.selectedNode}>
            <aside class="cg-aside surface-container-high medium-elevate padding no-border min" style="max-width: 30vw; min-width: 30rem">
              <div class="cg-header-row">
                <h2>{controls.selectedNode}</h2>
                <button class="link border" onClick={() => setControls("selectedNode", null)}>✕</button>
              </div>

              <Show when={selectedKind() === "hub" && selectedBin()}>
                {(_) => {
                  const bin = selectedBin()!; const years = [...bin.years].sort((a, b) => a - b); const topMax = bin.topNeighbours[0]?.freq ?? 1;
                  return (<>
                    <div class="bottom-padding">
                      <div>Events: {bin.eventCount}</div>
                      <div>Documents: {bin.docs.size}</div>
                      <div>Years: {years.length ? (years.length === 1 ? years[0] : `${ years[0] }–${ years[years.length - 1] }`) : "—"}</div>
                      <div>Hub connections: {graphData().nodes.find((n) => n.id === controls.selectedNode)?.hubDegree ?? 0}</div>
                    </div>
                    <h3 class="bottom-padding">Top neighbours</h3>
                    <div class="bottom-padding">
                      <For each={bin.topNeighbours.slice(0, MAX_TOP_N)}>
                        {(nb) => (<div class="cg-nb-row"><div class="cg-nb-bar-wrap"><div class="cg-nb-bar-fill hub" style={{ width: `${ (nb.freq / topMax) * 100 }%` }} /></div><span class="cg-nb-token">{nb.token}</span><span class="cg-nb-score">{nb.meanScore.toFixed(3)}</span></div>)}
                      </For>
                    </div>
                    <h3 class="bottom-padding">Sources</h3>
                    <Show when={selectedDocs().length > 0} fallback={<div class="error">No documents found</div>}>
                      <For each={selectedDocs()}>
                        {([docId, pubYear]) => (<button class="chip small-margin cg-chip-mono" onClick={() => showDocument(docId)}><span>{docId}</span><Show when={pubYear !== undefined}><span class="small-text"> {pubYear}</span></Show></button>)}
                      </For>
                    </Show>
                  </>);
                }}
              </Show>

              <Show when={selectedKind() === "event" && selectedEventNode()}>
                {(_) => {
                  const node = selectedEventNode()!; return (<>
                    <div class="bottom-padding">
                      <div>Token: {node.token ?? "—"}</div>
                      <div>Year: {node.pub_year ?? "—"}</div>
                      <div>token_idx: {node.token_idx ?? "—"}</div>
                      <Show when={node.doc_id}>
                        <div>
                          <button class="chip small-margin cg-chip-mono" onClick={() => showDocument(node.doc_id!)}><span>{node.doc_id}</span>
                          </button>
                        </div>
                      </Show>
                    </div>
                    <div class="bottom-padding small-text" style={{ opacity: 0.6 }}>Select a neighbour to see which sources share it.</div>
                  </>);
                }}
              </Show>

              <Show when={selectedKind() === "neighbour"}>
                <div class="bottom-padding"><div>Shared by {sharedByHubs().length} source(s)</div></div>
                <h3 class="bottom-padding">{controls.viewMode === "aggregated" ? "Hub contexts" : "Event contexts"}</h3>
                <Show when={sharedByHubs().length > 0} fallback={<div class="error">Not in any top-N list</div>}>
                  {(_) => {
                    const maxFreq = sharedByHubs()[0]?.freq ?? 1; return (
                      <div class="bottom-padding">
                        <For each={sharedByHubs()}>
                          {(h) => {
                            // In events view h.hub is a synthetic event node id —
                            // look up the node to get its token_idx and doc_id.
                            const sourceNode = () => controls.viewMode === "events"
                              ? graphData().nodes.find((n) => n.id === h.hub)
                              : null;
                            return (
                              <>
                                <article>
                                  <div class="row">
                                    <div class="cg-nb-bar-wrap">
                                      <div class="cg-nb-bar-fill neighbour" style={{ width: `${ (h.freq / maxFreq) * 100 }%` }} />
                                    </div>
                                    <span class="tooltip bottom">Mean score: {h.meanScore.toFixed(3)}</span>
                                  </div>


                                  <Show when={sourceNode()}>
                                    {(neighbourToken) => <>
                                      <span>{sourceNode()?.token ?? h.hub}</span>
                                      {" "}
                                      <span class="small-text" style={{ opacity: 0.6 }}>
                                        {neighbourToken().doc_id}/{neighbourToken().token_idx}
                                      </span>
                                      <EventContext open={true} docId={neighbourToken().doc_id!} tokenIdx={neighbourToken().token_idx!} />
                                    </>}
                                  </Show>
                                </article>
                              </>
                            );
                          }}
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
            <Show when={controls.viewMode === "aggregated"}><span class="cg-legend-hub" /> hubs ({graphData().nodes.filter((n) => n.kind === "hub").length})</Show>
            <Show when={controls.viewMode === "events"}><span class="cg-legend-event" /> events ({graphData().nodes.filter((n) => n.kind === "event").length})</Show>
            {" "}<span class="cg-legend-nb" /> neighbours ({graphData().nodes.filter((n) => n.kind === "neighbour").length})
            <Show when={controls.viewMode === "aggregated"}>{" • "}{graphData().hubHubEdges.length} similarity edges</Show>
            {" • "}{graphData().hubNbEdges.length} spokes
            {" • "}{filteredEvents().length} events
            <Show when={controls.fromYear !== yearBounds()[0] || controls.toYear !== yearBounds()[1]}>{" • "}{controls.fromYear}–{controls.toYear}</Show>
            {" • "}{totalEvents()} total
          </span>
        </footer>
      </div>
    </>
  );
};

export default CosmosComponent;