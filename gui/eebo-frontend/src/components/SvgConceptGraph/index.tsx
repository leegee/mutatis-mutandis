/**
 * ContextGraph.tsx
 *
 * Token-binned contextual similarity graph with neighbour expansion.
 *
 * VIEW MODES
 *  "aggregated"  — one hub node per distinct surface form (LAW, LAWES …)
 *                  aggregated across all events in the current year window.
 *                  Hubs linked by cosine similarity of neighbour-freq vectors.
 *                  Neighbour tokens expand as shared diamond nodes.
 *
 *  "events"      — one node per raw ConceptEvent, linked directly to their
 *                  top-N neighbour tokens.  Useful for inspecting individual
 *                  corpus contexts before aggregation collapses them.
 *
 * NODE KINDS
 *  "hub"       — one per distinct surface form aggregated across events.
 *                Radius ∝ sqrt(eventCount).  Filled circle.
 *                (aggregated mode only)
 *
 *  "event"     — one per raw ConceptEvent.
 *                Fixed small radius.  Filled circle.
 *                (events mode only)
 *
 *  "neighbour" — one per distinct neighbour token appearing in any hub's /
 *                event's top-N list.  Shared across sources: if PARLIAMENT
 *                appears in both LAW's and PREROGATIVE's top neighbours it is
 *                one node with two spokes.  Fixed small radius.  Diamond shape.
 *
 * EDGE KINDS
 *  "hub-hub"       — cosine similarity between two hubs' normalised
 *                    neighbour-frequency vectors.  Solid gradient line.
 *                    Only drawn when similarity ≥ minSimilarity.
 *                    Isolated hubs (no hub-hub edges) are still shown
 *                    because they carry spoke edges.
 *                    (aggregated mode only)
 *
 *  "hub-neighbour" — spoke from hub/event to each of its top-N neighbours.
 *                    Weight = normalised frequency (aggregated) or raw cosine
 *                    score (events).  Dashed, lower opacity.
 *
 * PIPELINE

 *   props.data (Tier2Data)
 *       │  filterByYearRange()
 *   ConceptEvent[]
 *       │
 *       ├─ [aggregated] aggregateByToken()  — bins events, normalised vectors
 *       │       │  buildContextualGraph(topN, minSimilarity, maxHubs)
 *       │       │    1. hub-hub edges: pairwise cosine ≥ minSimilarity
 *       │       │    2. neighbour nodes: union of top-N lists
 *       │       │    3. hub-neighbour spokes
 *       │       │    4. all isolated hubs retained
 *       │   ContextGraphData
 *       │
 *       └─ [events]     buildPureEventGraph(topN)
 *               │    1. one node per event
 *               │    2. hub-neighbour spokes to shared neighbour nodes
 *           ContextGraphData
 *       │  render()
 *   SVG — two edge layers, two node layers
 *

 * DRILL-DOWN

 *  Hub node    > event count, doc range, year range, source doc chips
 *  Event node  > doc_id, pub_year, neighbour list
 *  Neighbour   > "shared by" hub/event list (rhetorical coalition signal) +
 *                mean cosine score per source
 *

 * STAGE ARCHITECTURE  (ready for layers 2 + 3)

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

import './styles.css';
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../../corpus_config";
import type {
  ViewMode, ConceptEvent, ConceptData, TokenBin, ContextNode, HubHubEdge, HubNbEdge, AnyEdge, ContextGraphData, Tier2Data,
} from "./types";

const MAX_TOP_N = 20;
const hubSpread = () => 1;

const HUB_COLOR_LOW = "#5a87ba66";
const HUB_COLOR_HIGH = "#e9f3fcdd";
const EVENT_COLOR = "rgba(120,210,130,0.75)";
const NB_COLOR = "rgba(255,190,80,0.65)";
const NB_RADIUS = 5;
const DIAMOND_SIZE = 6;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [], hubHubEdges: [], hubNbEdges: [], allEdges: [],
  maxHubHubWeight: 1, maxEventCount: 1, maxHubDegree: 1,
};

//---------------------
// Data functions
//---------------------

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

/**
 * Aggregates contextual observations by lexical token.
 *
 * Events without a token field are skipped (no "__unknown__" fallback) so
 * that the hub set is clean.  Score sums are tracked separately from
 * frequency counts so meanScore is exact.
 */
function aggregateByToken(events: ConceptEvent[]): Map<string, TokenBin> {
  const bins = new Map<string, TokenBin>();

  for (const event of events) {
    const binKey = event.token;
    // Skip events with no token — don't pollute hub set with unknowns.
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
      bin.neighbourScoreSum.set(nb.token, (bin.neighbourScoreSum.get(nb.token) ?? 0) + nb.score);
    }
  }

  // Normalise frequencies and compute top neighbours.
  for (const bin of bins.values()) {
    const total = [...bin.neighbourFreq.values()].reduce((a, b) => a + b, 0);

    const normFreq = new Map<string, number>();
    for (const [tok, count] of bin.neighbourFreq)
      normFreq.set(tok, total > 0 ? count / total : 0);
    bin.neighbourFreq = normFreq;

    bin.topNeighbours = [...normFreq.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 30)   // store top-30 so the drill-down has headroom
      .map(([tok, freq]) => {
        const rawCount = Math.round(freq * total); // recover raw count for division
        return {
          token: tok,
          freq,
          meanScore: rawCount > 0
            ? (bin.neighbourScoreSum.get(tok) ?? 0) / rawCount
            : 0,
        };
      });
  }

  return bins;
}

/**
 * Sparse cosine similarity over neighbour-frequency vectors.
 * Iterates the smaller map for efficiency.
 */
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
 * Build the aggregated two-kind graph (hub + neighbour nodes).
 */
function buildContextualGraph(
  bins: Map<string, TokenBin>,
  topN: number,
  minSimilarity: number,
  maxHubs: number,
): ContextGraphData {
  const hubKeys = [...bins.keys()];
  if (hubKeys.length === 0) return EMPTY_GRAPH;

  // 1. Hub-hub edges: pairwise cosine -------------------------------------
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

  // Hub degree from hub-hub edges only.
  const hubHubDegree = new Map<string, number>();
  for (const [a, b] of rawHubHub) {
    hubHubDegree.set(a, (hubHubDegree.get(a) ?? 0) + 1);
    hubHubDegree.set(b, (hubHubDegree.get(b) ?? 0) + 1);
  }

  // Retain all hubs but prefer the most contextually connected ones when
  // truncating: sort by hub-hub degree desc, then eventCount desc.
  const sortedHubs = hubKeys
    .sort((a, b) => {
      const dd = (hubHubDegree.get(b) ?? 0) - (hubHubDegree.get(a) ?? 0);
      if (dd !== 0) return dd;
      return bins.get(b)!.eventCount - bins.get(a)!.eventCount;
    })
    .slice(0, maxHubs);

  const hubSet = new Set(sortedHubs);

  // 2. Hub nodes---
  const nodeMap = new Map<string, ContextNode>();

  for (const key of sortedHubs) {
    nodeMap.set(key, {
      id: key,
      kind: "hub",
      eventCount: bins.get(key)!.eventCount,
      hubDegree: hubHubDegree.get(key) ?? 0,
      degree: hubHubDegree.get(key) ?? 0,
    });
  }

  // 3. Neighbour nodes + hub-neighbour edges -------------------------------
  const spokeTriples: Array<[string, string, number]> = [];

  for (const hubKey of sortedHubs) {
    const bin = bins.get(hubKey)!;
    for (const nb of bin.topNeighbours.slice(0, topN)) {
      if (!nodeMap.has(nb.token)) {
        nodeMap.set(nb.token, {
          id: nb.token, kind: "neighbour",
          eventCount: 0, hubDegree: 0, degree: 0,
        });
      }
      spokeTriples.push([hubKey, nb.token, nb.freq]);
    }
  }

  for (const [hubKey, nbToken] of spokeTriples) {
    nodeMap.get(hubKey)!.degree += 1;
    nodeMap.get(nbToken)!.degree += 1;
  }

  const nodes = [...nodeMap.values()];

  // 4. Materialise edges ---------------------------------------------------
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

/**
 * Build a raw event graph: one node per ConceptEvent, linked to their top-N
 * neighbour tokens.  Neighbour nodes are shared across events.
 * No hub-hub edges.
 */
function buildPureEventGraph(
  events: ConceptEvent[],
  topN: number,
): ContextGraphData {
  if (events.length === 0) return EMPTY_GRAPH;

  const nodeMap = new Map<string, ContextNode>();
  const hubNbEdges: HubNbEdge[] = [];

  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const nodeId = event.event_id !== undefined
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

    const top = [...event.neighbours]
      .sort((a, b) => b.score - a.score)
      .slice(0, topN);

    for (const nb of top) {
      if (!nodeMap.has(nb.token)) {
        nodeMap.set(nb.token, {
          id: nb.token, kind: "neighbour",
          eventCount: 0, hubDegree: 0, degree: 0,
        });
      }
      const nbNode = nodeMap.get(nb.token)!;
      hubNbEdges.push({
        kind: "hub-neighbour" as const,
        source: eventNode,
        target: nbNode,
        weight: nb.score,
      });
      eventNode.degree += 1;
      nbNode.degree += 1;
    }
  }

  const nodes = [...nodeMap.values()];

  console.log("[ctx-graph/events] event nodes:", events.length,
    "| nb nodes:", nodes.filter(n => n.kind === "neighbour").length,
    "| spoke edges:", hubNbEdges.length);

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


const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

export interface Props {
  data: Tier2Data;
}

const ContextGraph5: Component<Props> = (props) => {
  const concepts = Object.keys(props.data as Tier2Data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [viewMode, setViewMode] = createSignal<ViewMode>("aggregated");
  const [maxHubs, setMaxHubs] = createSignal(50);
  const [topN, setTopN] = createSignal(5);
  const [minSimilarity, setMinSimilarity] = createSignal(0.5);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);
  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);
  const [yearMode, setYearMode] = createSignal<"single" | "range">("single");

  //  Year bounds

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

  // Filtered events

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    const events = cd.events;
    return fromYear() <= min && toYear() >= max
      ? events
      : filterByYearRange(events, fromYear(), toYear());
  });

  // Aggregation (aggregated mode only) ------------------------------------

  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(yearFiltered())
  );

  // Graph----------

  const graphData = createMemo<ContextGraphData>(() =>
    viewMode() === "events"
      ? buildPureEventGraph(yearFiltered(), topN())
      : buildContextualGraph(tokenBins(), topN(), minSimilarity(), maxHubs())
  );

  // Drill-down-----

  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
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

  // Event drill-down: the raw ContextNode (carries token, doc_id, pub_year)
  const selectedEventNode = createMemo<ContextNode | null>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find(n => n.id === id) ?? null;
  });

  // Neighbour drill-down: which hubs/events share this token
  const sharedByHubs = createMemo<Array<{ hub: string; freq: number; meanScore: number }>>(() => {
    const id = selectedNode();
    if (!id || selectedKind() !== "neighbour") return [];

    if (viewMode() === "aggregated") {
      const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
      for (const [hubKey, bin] of tokenBins()) {
        const nb = bin.topNeighbours.find(n => n.token === id);
        if (nb) result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore });
      }
      return result.sort((a, b) => b.freq - a.freq);
    }

    // Events mode: find events that have this token in their top-N spokes.
    const result: Array<{ hub: string; freq: number; meanScore: number }> = [];
    for (const edge of graphData().hubNbEdges) {
      if ((edge.target as ContextNode).id === id) {
        const src = edge.source as ContextNode;
        result.push({ hub: src.doc_id ?? src.id, freq: edge.weight, meanScore: edge.weight });
      }
    }
    return result.sort((a, b) => b.freq - a.freq);
  });

  // D3 render-----

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
        .text("No graph: try reducing min similarity or increasing top N");
      return;
    }

    const hubRadius = d3.scaleSqrt().domain([0, maxEventCount]).range([8, 40]);
    const hubColor = d3.scaleLinear<string>()
      .domain([0, Math.max(1, maxHubDegree)])
      .range([HUB_COLOR_LOW, HUB_COLOR_HIGH]);
    const hhOpacity = d3.scaleLinear().domain([0, maxHubHubWeight]).range([0.25, 0.85]);
    const hhWidth = d3.scaleLinear().domain([0, maxHubHubWeight]).range([1, 7]);
    const spokeOpacity = d3.scaleLinear().domain([0, 1]).range([0.5, 0.95]);
    const spokeWidth = d3.scaleLinear().domain([0, 1]).range([1, 4]);

    const container = svg.append("g").attr("class", "zoom-container");
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (ev) => container.attr("transform", ev.transform))
    );

    const defs = container.append("defs");

    // Gradient defs for hub-hub edges.
    hubHubEdges.forEach((d, i) => {
      const grad = defs.append("linearGradient")
        .attr("id", `hh-${ i }`).attr("gradientUnits", "userSpaceOnUse");
      grad.append("stop").attr("offset", "0%")
        .attr("stop-color", hubColor(d.source.hubDegree));
      grad.append("stop").attr("offset", "100%")
        .attr("stop-color", hubColor(d.target.hubDegree));
    });

    // Edge layer 1: hub-neighbour spokes ------------------------------------
    const spokeSelection = container.append("g").attr("class", "spokes")
      .selectAll<SVGLineElement, HubNbEdge>("line")
      .data(hubNbEdges).join("line")
      .attr("stroke", NB_COLOR)
      .attr("stroke-opacity", (d) => spokeOpacity(d.weight))
      .attr("stroke-width", (d) => spokeWidth(d.weight))
      .attr("stroke-dasharray", "3 3");

    // Edge layer 2: hub-hub similarity edges --------------------------------
    const hhSelection = container.append("g").attr("class", "hh-edges")
      .selectAll<SVGLineElement, HubHubEdge>("line")
      .data(hubHubEdges).join("line")
      .attr("stroke", (_, i) => `url(#hh-${ i })`)
      .attr("stroke-opacity", (d) => hhOpacity(d.weight))
      .attr("stroke-width", (d) => hhWidth(d.weight));

    // Neighbour nodes -------------------------------------------------------
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

    nbGroup.append("rect")
      .attr("x", -DIAMOND_SIZE).attr("y", -DIAMOND_SIZE)
      .attr("width", DIAMOND_SIZE * 2).attr("height", DIAMOND_SIZE * 2)
      .attr("transform", "rotate(45)")
      .attr("fill", NB_COLOR)
      .attr("stroke", "rgba(255,220,120,0.3)")
      .attr("stroke-width", 1);

    nbGroup.append("text")
      .text(d => d.id)
      .attr("dx", DIAMOND_SIZE + 4).attr("dy", "0.35em")
      .attr("text-anchor", "start")
      .attr("font-size", "10pt")
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", "rgba(255,220,140,0.85)")
      .attr("pointer-events", "none");

    // Hub / event nodes (on top) -------------------------------------------
    const sourceNodes = nodes.filter(n => n.kind === "hub" || n.kind === "event");
    const sourceGroup = container.append("g").attr("class", "hub-nodes")
      .selectAll<SVGGElement, ContextNode>("g")
      .data(sourceNodes, d => d.id).join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode(prev => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGGElement, ContextNode>()
          .on("start", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
          .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
          .on("end", (ev, d) => { if (!ev.active) simulationRef?.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    sourceGroup.append("circle")
      .attr("r", d => d.kind === "hub" ? hubRadius(d.eventCount) : 6)
      .attr("fill", d => d.kind === "hub" ? hubColor(d.hubDegree) : EVENT_COLOR)
      .attr("stroke", d =>
        d.kind === "hub" ? "rgba(200,230,255,0.3)" : "rgba(180,255,190,0.3)"
      )
      .attr("stroke-width", 1.5);

    // Event-count badge (hub only, when large enough).
    sourceGroup.append("text")
      .text(d => d.kind === "hub" && hubRadius(d.eventCount) > 12 ? String(d.eventCount) : "")
      .attr("dy", "0.35em")
      .attr("text-anchor", "middle")
      .attr("font-size", "12pt")
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", "rgba(255,255,255,0.5)")
      .attr("pointer-events", "none");

    // Label below circle.
    sourceGroup.append("text")
      .text(d => d.kind === "hub" ? d.id : (d.token ?? d.id))
      .attr("dy", d => (d.kind === "hub" ? hubRadius(d.eventCount) : 6) + 13)
      .attr("text-anchor", "middle")
      .attr("font-size", d =>
        d.kind === "hub"
          ? Math.max(9, Math.min(13, hubRadius(d.eventCount) * 0.7)) + "pt"
          : "8pt"
      )
      .attr("font-family", "'IBM Plex Mono','Courier New',monospace")
      .attr("fill", d =>
        d.kind === "hub" ? "rgba(210,235,255,0.9)" : "rgba(180,255,190,0.8)"
      )
      .attr("pointer-events", "none");

    // Tooltip-------
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

    sourceGroup
      .on("mouseenter", (ev, d) => {
        if (d.kind === "hub") {
          const bin = tokenBins().get(d.id);
          const years = bin ? [...bin.years].sort((a, b) => a - b) : [];
          const yStr = years.length
            ? `${ years[0] }${ years.length > 1 ? `–${ years[years.length - 1] }` : "" }`
            : "—";
          showTip(ev,
            `<aside><h6 class="bottom-padding">${ d.id }</h6>` +
            `Events: ${ d.eventCount }<br/>Connections: ${ d.hubDegree }<br/>` +
            `Documents: ${ bin?.docs.size ?? "—" }<br/>Years: ${ yStr }</aside>`);
        } else {
          showTip(ev,
            `<aside><h6 class="bottom-padding">${ d.token ?? d.id }</h6>` +
            `Doc: ${ d.doc_id ?? "—" }<br/>Year: ${ d.pub_year ?? "—" }</aside>`);
        }
      })
      .on("mousemove", moveTip).on("mouseleave", hideTip);

    nbGroup
      .on("mouseenter", (ev, d) => {
        const hubs = sharedByHubs();
        const lines = hubs.length
          ? hubs.slice(0, 5).map(h => `${ h.hub } (${ h.freq.toFixed(3) })`).join("<br/>")
          : "—";
        showTip(ev,
          `<aside><h6 class="bottom-padding">${ d.id }</h6>` +
          `Shared by ${ d.degree } source(s):<br/>${ lines }</aside>`);
      })
      .on("mousemove", moveTip).on("mouseleave", hideTip);

    // Simulation----
    if (simulationRef) simulationRef.stop();

    simulationRef = d3.forceSimulation<ContextNode>(nodes)
      .force("link",
        d3.forceLink<ContextNode, AnyEdge>(allEdges)
          .id(d => d.id)
          // .distance(d => d.kind === "hub-hub" ? Math.max(80, 260 - (d as HubHubEdge).weight * 180) : 60 )
          .distance(d =>
            d.kind === "hub-hub"
              ? Math.max(40, (260 - (d as HubHubEdge).weight * 180) * hubSpread())
              : 60
          )
          .strength(d => d.kind === "hub-hub" ? 0.55 : 0.8)
      )
      .force("charge", d3.forceManyBody()
        // .strength((d) => d.kind === "hub" ? -280 : -40)
        .strength(d => d.kind === "hub" ? -280 * hubSpread() : -40)
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

        sourceGroup.attr("transform", d => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`);
        nbGroup.attr("transform", d => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`);
      });
  }

  createEffect(() => { graphData(); if (svgRef) render(); });
  onCleanup(() => {
    simulationRef?.stop();
    d3.select("body").selectAll(".cg-tooltip").remove();
  });

  return (
    <div class="svg-cg-layout">

      <header class="center-align fill max surface-container-low small-padding top-padding">
        <nav>

          <div class="field suffix border middle-align">
            <select value={concept()}
              onChange={(e) => { setConcept(e.currentTarget.value); setSelectedNode(null); }}>
              <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
            </select>
            <output>Concept</output>
          </div>

          {/* View mode toggle */}
          <div class="field suffix border middle-align">
            <select value={viewMode()}
              onChange={(e) => { setViewMode(e.currentTarget.value as ViewMode); setSelectedNode(null); }}>
              <option value="aggregated">Aggregated</option>
              <option value="events">Events</option>
            </select>
            <output>View</output>
          </div>

          {/* Max hubs: aggregated mode only */}
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

          {/* Min similarity: aggregated mode only */}
          <Show when={viewMode() === "aggregated"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input type="range" min={0.01} max={0.95} step={0.05} value={minSimilarity()}
                  onInput={(e) => setMinSimilarity(Number(e.currentTarget.value))} />
                <span /><span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                Min similarity {minSimilarity().toFixed(2)}
              </output>
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

            {/* -- Hub drill-down -- */}
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
                      <For each={bin.topNeighbours.slice(0, MAX_TOP_N)}>
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

            {/* -- Event drill-down -- */}
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
                          <button class="chip small-margin cg-chip-mono"
                            onClick={() => showDocument(node.doc_id!)}>
                            <span>{node.doc_id}</span>
                          </button>
                        </div>
                      </Show>
                    </div>
                    <div class="bottom-padding small-text" style={{ opacity: 0.6 }}>
                      Select a neighbour diamond to see which sources share it.
                    </div>
                  </>
                );
              }}
            </Show>

            {/* -- Neighbour drill-down -- */}
            <Show when={selectedKind() === "neighbour"}>
              <div class="bottom-padding">
                <div>Shared by {sharedByHubs().length} source(s)</div>
              </div>
              <h3 class="bottom-padding">
                {viewMode() === "aggregated" ? "Hub contexts" : "Event contexts"}
              </h3>
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
  );
};

export default ContextGraph5;
