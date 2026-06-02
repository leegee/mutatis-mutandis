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

import type { ContextGraphData, ContextNode, TokenBin } from "./types";

import { controls, setControls } from "../../state/controls.store";
import {
  aggregateByToken,
  buildContextualGraph,
  buildPureEventGraph,
} from "../../lib/contextGraphUtils";
import { getYearFiltered } from "../../state/selectors";

interface GraphCanvasProps {
  graphData: ContextGraphData;
  viewMode: "events" | "aggregated";
  hubSpread: number;
  selectedNode: string | null;
  showEventLabels: boolean;
}

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

const HUB_COLOR_LOW_RGBA: [number, number, number, number] = [
  0.35, 0.53, 0.73, 0.4,
];
const HUB_COLOR_HIGH_RGBA: [number, number, number, number] = [
  0.91, 0.95, 0.99, 0.87,
];
const EVENT_RGBA: [number, number, number, number] = [0.47, 0.82, 0.51, 0.75];
const NB_RGBA: [number, number, number, number] = [1.0, 0.75, 0.31, 0.65];
const HH_LINK_BASE_RGBA: [number, number, number, number] = [
  0.55, 0.75, 0.95, 0.95,
];
const SPOKE_LINK_RGBA: [number, number, number, number] = [
  1.0, 0.75, 0.31, 0.9,
];

const HH_WIDTH_MIN = 2;
const HH_WIDTH_MAX = 3;
const HH_ALPHA_MIN = 0.75;
const HH_ALPHA_MAX = 1;
const SPOKE_ALPHA_MIN = 0.75;
const SPOKE_ALPHA_MAX = 0.99;
const SPOKE_WIDTH_MIN = 2;
const SPOKE_WIDTH_MAX = 6;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [],
  hubHubEdges: [],
  hubNbEdges: [],
  allEdges: [],
  maxHubHubWeight: 1,
  maxEventCount: 1,
  maxHubDegree: 1,
};

interface WorldNode extends ContextNode {
  cosmosIndex: number;
}

class GraphWorld {
  private cosmos: Graph;
  private nodeRegistry = new Map<string, WorldNode>();
  private reverseIndex = new Map<number, WorldNode>();
  private nextIndex = 0;

  private maxNodes = 25000;
  private maxEdges = 150000;

  private persistentPositions = new Map<string, [number, number]>();
  private pointColors = new Float32Array(this.maxNodes * 4);
  private pointSizes = new Float32Array(this.maxNodes);

  private links = new Float32Array(this.maxEdges * 2);
  private linkColors = new Float32Array(this.maxEdges * 4);
  private linkWidths = new Float32Array(this.maxEdges);

  private scales = {
    hubColor: null as any,
    hubSize: null as any,
    hhAlpha: null as any,
    hhWidth: null as any,
    spokeAlpha: null as any,
    spokeWidth: null as any,
  };

  constructor(cosmos: Graph) {
    this.cosmos = cosmos;
  }

  initialize(gd: ContextGraphData) {
    this.nodeRegistry.clear();
    this.reverseIndex.clear();
    // this.persistentPositions.clear();
    this.nextIndex = 0;

    this.rebuildScales(gd);
    this.assignPersistentIndices(gd);
    this.initializeRandomPositions(gd);

    this.updateTimeSlice(gd, true);
  }

  updateTimeSlice(gd: ContextGraphData, fullReset = false) {
    this.rebuildScales(gd);

    const nodeCount = gd.nodes.length;
    const edgeCount = gd.allEdges.length;

    // NODES
    for (let i = 0; i < nodeCount; i++) {
      const cn = gd.nodes[i];
      const wn = this.nodeRegistry.get(cn.id);
      if (!wn) continue;

      const idx = wn.cosmosIndex;

      let rgba: [number, number, number, number];
      let size: number;

      if (cn.kind === "hub") {
        rgba = this.scales.hubColor(cn.hubDegree);
        size = this.scales.hubSize(cn.eventCount);
      } else if (cn.kind === "event") {
        rgba = EVENT_RGBA;
        size = EVENT_SIZE;
      } else {
        rgba = NB_RGBA;
        size = NB_POINT_SIZE;
      }

      const base = idx * 4;
      this.pointColors[base] = rgba[0];
      this.pointColors[base + 1] = rgba[1];
      this.pointColors[base + 2] = rgba[2];
      this.pointColors[base + 3] = rgba[3];
      this.pointSizes[idx] = size;
    }

    // EDGES
    let linkIdx = 0;

    const zoom = this.cosmos.getZoomLevel?.() ?? 1;
    const zoomFactor = Math.max(1, 2 / zoom);
    const zoomAlphaBoost = Math.min(1, zoom * 0.8);

    for (let i = 0; i < edgeCount; i++) {
      const edge = gd.allEdges[i];

      const src = this.nodeRegistry.get(edge.sourceId);
      const tgt = this.nodeRegistry.get(edge.targetId);

      if (!src || !tgt) continue;

      this.links[linkIdx * 2] = src.cosmosIndex;
      this.links[linkIdx * 2 + 1] = tgt.cosmosIndex;

      let alpha: number;
      let baseColor: [number, number, number, number];
      let width: number;

      if (edge.kind === "hub-hub") {
        alpha = Math.max(0.25, this.scales.hhAlpha(edge.weight));
        baseColor = HH_LINK_BASE_RGBA;
        width = this.scales.hhWidth(edge.weight);
      } else {
        alpha = Math.max(0.25, this.scales.spokeAlpha(edge.weight));
        baseColor = SPOKE_LINK_RGBA;
        width = this.scales.spokeWidth(edge.weight);
      }

      // RGBA
      this.linkColors[linkIdx * 4] = baseColor[0];
      this.linkColors[linkIdx * 4 + 1] = baseColor[1];
      this.linkColors[linkIdx * 4 + 2] = baseColor[2];
      this.linkColors[linkIdx * 4 + 3] = alpha * zoomAlphaBoost;

      // width (zoom-aware)
      this.linkWidths[linkIdx] = width * zoomFactor;

      linkIdx++;
    }

    // COMMIT TO COSMOS
    this.cosmos.setPointColors(this.pointColors.subarray(0, nodeCount * 4));
    this.cosmos.setPointSizes(this.pointSizes.subarray(0, nodeCount));

    this.cosmos.setLinks(this.links.subarray(0, linkIdx * 2));
    this.cosmos.setLinkColors(this.linkColors.subarray(0, linkIdx * 4));
    this.cosmos.setLinkWidths(this.linkWidths.subarray(0, linkIdx));

    if (fullReset) {
      this.cosmos.setPointPositions(this.getActivePositions(gd));
      this.cosmos.trackPointPositionsByIndices(gd.nodes.map((_, idx) => idx));
    }

    this.cosmos.render();
  }

  private assignPersistentIndices(gd: ContextGraphData) {
    gd.nodes.forEach((cn) => {
      if (!this.nodeRegistry.has(cn.id)) {
        const wn: WorldNode = { ...cn, cosmosIndex: this.nextIndex++ };
        this.nodeRegistry.set(cn.id, wn);
        this.reverseIndex.set(wn.cosmosIndex, wn);
      }
    });
  }

  private initializeRandomPositions(gd: ContextGraphData) {
    gd.nodes.forEach((cn) => {
      if (!this.persistentPositions.has(cn.id)) {
        this.persistentPositions.set(cn.id, [
          SPACE / 2 + (Math.random() - 0.5) * 50,
          SPACE / 2 + (Math.random() - 0.5) * 50,
        ]);
      }
    });
  }

  private getActivePositions(gd: ContextGraphData): Float32Array {
    const nodeCount = gd.nodes.length;
    const active = new Float32Array(nodeCount * 2);

    gd.nodes.forEach((cn, i) => {
      const pos = this.persistentPositions.get(cn.id)!;
      active[i * 2] = pos[0];
      active[i * 2 + 1] = pos[1];
    });

    return active;
  }

  private rebuildScales(gd: ContextGraphData) {
    this.scales.hubColor = d3
      .scaleLinear<[number, number, number, number]>()
      .domain([0, Math.max(1, gd.maxHubDegree)])
      .range([HUB_COLOR_LOW_RGBA, HUB_COLOR_HIGH_RGBA]);

    this.scales.hubSize = d3
      .scaleSqrt()
      .domain([0, gd.maxEventCount])
      .range([HUB_MIN_SIZE, HUB_MAX_SIZE]);

    this.scales.hhAlpha = d3
      .scaleLinear()
      .domain([0, gd.maxHubHubWeight])
      .range([HH_ALPHA_MIN, HH_ALPHA_MAX]);

    this.scales.hhWidth = d3
      .scaleLinear()
      .domain([0, gd.maxHubHubWeight])
      .range([HH_WIDTH_MIN, HH_WIDTH_MAX]);

    this.scales.spokeAlpha = d3
      .scaleLinear()
      .domain([0, 1])
      .range([SPOKE_ALPHA_MIN, SPOKE_ALPHA_MAX]);

    this.scales.spokeWidth = d3
      .scalePow()
      .exponent(0.25)
      .domain([0, 1])
      .range([SPOKE_WIDTH_MIN, SPOKE_WIDTH_MAX]);
  }

  getNodeByIndex(idx: number): WorldNode | undefined {
    return this.reverseIndex.get(idx);
  }

  getNodeById(id: string): WorldNode | undefined {
    return this.nodeRegistry.get(id);
  }
}

const GraphCanvas: Component<GraphCanvasProps> = (props: GraphCanvasProps) => {
  const [labelPositions, setLabelPositions] = createSignal<
    Array<{ id: string; label: string; kind: string; x: number; y: number }>
  >([]);
  const [hoveredId, setHoveredId] = createSignal<string | null>(null);
  const [simulating, setSimulating] = createSignal(false);
  let simTimeoutHandle = 0;
  let simGeneration = 0;

  function stopSimulating(generation: number) {
    if (generation !== simGeneration) return;
    clearTimeout(simTimeoutHandle);
    setSimulating(false);
    cosmosGraph?.pause();
    cosmosGraph?.fitView();
  }

  function startSimulating(nodeCount = 0) {
    clearTimeout(simTimeoutHandle);
    const generation = ++simGeneration;
    setSimulating(true);
    const ms = Math.min(12_000, Math.max(1_500, nodeCount * 10));
    simTimeoutHandle = window.setTimeout(() => stopSimulating(generation), ms);
  }

  const [filteredEventsResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredEvents = () => filteredEventsResource() ?? [];

  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(filteredEvents()),
  );

  let wrapRef!: HTMLDivElement;
  let cosmosGraph: Graph | null = null;
  let world: GraphWorld | null = null;
  let rafHandle = 0;
  let mouseClientX = 0;
  let mouseClientY = 0;
  let tooltipEl: HTMLDivElement | null = null;

  const getTooltip = (): HTMLDivElement => {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className =
        "cg-tooltip surface-container-highest border large-elevate padding";
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  };

  function showTooltip(wn: WorldNode) {
    const tip = getTooltip();
    let html = "";
    if (wn.kind === "hub") {
      const bin = untrack(tokenBins).get(wn.id);
      const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
      const yStr = years.length
        ? `${years[0]}${years.length > 1 ? `–${years[years.length - 1]}` : ""}`
        : "-";

      html = `<aside>
        <h6 class="bottom-padding">${wn.id}</h6>
        Events: ${wn.eventCount}<br/>
        Connections: ${wn.hubDegree}<br/>
        Documents: ${bin?.docs.size ?? "-"}<br/>
        Years: ${yStr}
      </aside>`;
    } else if (wn.kind === "event") {
      html = `<aside>
        <h6 class="bottom-padding"><q>${wn.token ?? wn.id}</q></h6>
        Year: ${wn.pub_year ?? "-"}<br/>
        Doc: ${wn.doc_id ?? "-"}<br/>
        token_idx: ${wn.token_idx ?? "-"}
        </aside>`;
    } else {
      const id = wn.id;
      const viewMode = untrack(() => controls.viewMode);
      const bins = untrack(tokenBins);
      const gd = props.graphData;
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
      html = `<aside>
        <h6 class="bottom-padding">${wn.id}</h6>
        Shared by ${wn.degree} source(s):<br/>
        ${
          hubs.length
            ? hubs
                .slice(0, 5)
                .map((h) => `${h.hub} (${h.freq.toFixed(3)})`)
                .join("<br/>")
            : "-"
        }
      </aside>`;
    }
    tip.innerHTML = html;
    Object.assign(tip.style, {
      opacity: "1",
      left: `${mouseClientX + 14}px`,
      top: `${mouseClientY - 10}px`,
    });
  }

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
        const showEventLabels = controls.showEventLabels;
        let dirty = posMap.size !== prevLabelCache.size;
        const nextLabels: Array<{
          id: string;
          label: string;
          kind: string;
          x: number;
          y: number;
        }> = [];

        posMap.forEach(([sx, sy], idx) => {
          const wn = world!.getNodeByIndex(idx);
          if (!wn) return;
          if (wn.kind === "hub" && wn.id === currentConcept && wn.id !== hid)
            return;
          if (wn.kind === "event" && !showEventLabels && wn.id !== hid) return;

          const [x, y] = cosmosGraph!.spaceToScreenPosition([sx, sy]);
          const label = wn.kind === "event" ? (wn.token ?? wn.id) : wn.id;
          nextLabels.push({ id: wn.id, label, kind: wn.kind, x, y });

          if (!dirty) {
            const prev = prevLabelCache.get(idx);
            if (
              !prev ||
              Math.abs(prev.x - x) > LABEL_MOVE_THRESHOLD ||
              Math.abs(prev.y - y) > LABEL_MOVE_THRESHOLD
            )
              dirty = true;
          }
        });

        if (dirty) {
          prevLabelCache.clear();
          for (const lbl of nextLabels) {
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
      hoveredPointRingColor: "aqua",
      renderHoveredPointRing: true,
      onSimulationEnd: () => stopSimulating(simGeneration),
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

  // Major updates (concept, view mode, topN, etc.)
  createEffect(() => {
    const gd = props.graphData;
    if (!wrapRef) return;

    ensureCosmosInitialised();
    setLabelPositions([]);

    if (gd.nodes.length === 0) {
      cosmosGraph!.setPointPositions(new Float32Array(0));
      cosmosGraph!.setPointColors(new Float32Array(0));
      cosmosGraph!.setPointSizes(new Float32Array(0));
      cosmosGraph!.setLinks(new Float32Array(0));
      cosmosGraph!.setLinkColors(new Float32Array(0));
      cosmosGraph!.setLinkWidths(new Float32Array(0));
      cosmosGraph!.render();
      return;
    }

    const hubCount = gd.nodes.filter(
      (n) => n.kind === "hub" || n.kind === "event",
    ).length;
    const isDegenerate =
      gd.allEdges.length === 0 || gd.nodes.length <= 1 || hubCount <= 1;

    cosmosGraph!.setConfig({
      spaceSize: SPACE,
      simulationRepulsion: isDegenerate
        ? 0
        : BASE_REPULSION * (0.6 + controls.hubSpread * 0.4),
      simulationLinkSpring: isDegenerate ? 0 : BASE_LINK_SPRING,
      simulationGravity: isDegenerate ? 0.8 : BASE_GRAVITY,
      simulationFriction: BASE_FRICTION,
      enableDrag: true,
      fitViewPadding: FIT_VIEW_PADDING,
    });

    if (world) {
      world.initialize(gd);
      startSimulating(gd.nodes.length);
      cosmosGraph!.start();
    }
  });

  // Fast timeline updates
  createEffect(() => {
    const gd = props.graphData;
    if (!world || gd.nodes.length === 0) return;
    world.updateTimeSlice(gd, false);
  });

  onCleanup(() => {
    cancelAnimationFrame(rafHandle);
    clearTimeout(simTimeoutHandle);
    cosmosGraph?.pause();
    tooltipEl?.remove();
    tooltipEl = null;
  });

  // function forceRedraw() {
  //   if (!cosmosGraph || !world) return;
  //   const gd = props.graphData;
  //   world.initialize(gd);
  //   cosmosGraph.start();
  // }

  return (
    <>
      <div
        ref={wrapRef!}
        class="cg-canvas-wrap surface-container"
        style={{ visibility: simulating() ? "hidden" : "visible" }}
      >
        <div class="cg-labels">
          <For each={labelPositions()}>
            {(lbl) => (
              <span
                class={`cg-label ${lbl.kind}`}
                style={{
                  left: `${lbl.x}px`,
                  top: `${lbl.y - (lbl.kind == "hub" ? 28 : 24)}px`,
                }}
              >
                {lbl.label}
              </span>
            )}
          </For>
        </div>

        <Show when={props.graphData.nodes.length === 0}>
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              "align-items": "center",
              "justify-content": "center",
              "pointer-events": "none",
              "z-index": 999,
            }}
          >
            <span class="padding error fade-in-1s">
              <h4>
                No data to graph for
                <span style="text-transform:capitalize">
                  "{controls.concept.toLocaleLowerCase()}"
                </span>{" "}
                from {controls.fromYear}
              </h4>
              <p>
                Try reducing the minimum similarity threshold, or increasing the
                value of top N, or changing the year you are viewing.
              </p>
            </span>
          </div>
        </Show>
      </div>

      <Show when={simulating()}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            "flex-direction": "column",
            "align-items": "center",
            "justify-content": "center",
            gap: "1rem",
            "pointer-events": "none",
          }}
        >
          <h3>Settling layout…</h3>
          <progress class="circle light-green-text" />
        </div>
      </Show>
    </>
  );
};

export default GraphCanvas;
