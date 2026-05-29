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

 *   tier2Data (Tier2Data)
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
  createMemo,
  createEffect,
  onCleanup,
  For,
  Show,
  type Component,
} from "solid-js";

import * as d3 from "d3";

import './styles.css';
import { tier2Data } from "../../state/tier2data.store";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../../corpus_config";
import type { TokenBin, ContextNode, HubHubEdge, HubNbEdge, AnyEdge, ContextGraphData } from "../../types/context-graph.types";
import ControlsHeader from "../ControlsHeader";
import { controls, setControls } from "../../state/controls";
import { aggregateByToken, buildContextualGraph, buildPureEventGraph, filterByYearRange, scanYearRange } from "../../lib/contextGraphUtils";

const MAX_TOP_N = 20;

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

const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");


const ContextGraph5: Component = () => {
  const concepts = Object.keys(tier2Data)

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = tier2Data[controls.concept];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });


  const yearFiltered = createMemo(() => {
    const cd = tier2Data[controls.concept];
    if (!cd) return [];
    const [min, max] = yearBounds();
    const events = cd.events;
    return controls.fromYear <= min && controls.toYear >= max
      ? events
      : filterByYearRange(events, controls.fromYear, controls.toYear);
  });


  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(yearFiltered())
  );


  const graphData = createMemo<ContextGraphData>(() =>
    controls.viewMode === "events"
      ? buildPureEventGraph(yearFiltered(), controls.topN, EMPTY_GRAPH)
      : buildContextualGraph(tokenBins(), controls.topN, controls.minSimilarity, controls.maxHubs, EMPTY_GRAPH)
  );


  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
    const id = controls.selectedNode;
    if (!id) return null;
    return graphData().nodes.find(n => n.id === id)?.kind ?? null;
  });


  // Hub drill-down: the TokenBin
  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = controls.selectedNode;
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
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find(n => n.id === id) ?? null;
  });

  // Neighbour drill-down: which hubs/events share this token
  const sharedByHubs = createMemo<Array<{ hub: string; freq: number; meanScore: number }>>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "neighbour") return [];

    if (controls.viewMode === "aggregated") {
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

  type EdgeKey = string;

  const key = (a: ContextNode, b: ContextNode): EdgeKey =>
    a.id < b.id ? `${ a.id }__${ b.id }` : `${ b.id }__${ a.id }`;

  function render() {
    const { nodes, hubHubEdges, hubNbEdges, allEdges,
      maxHubHubWeight, maxEventCount, maxHubDegree } = graphData();
    const svg = d3.select(svgRef);
    const W = svgRef.clientWidth;
    const H = svgRef.clientHeight;

    svg.selectAll("*").remove();
    d3.select("body").selectAll(".cg-tooltip").remove();

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
    hubHubEdges.forEach((d: HubHubEdge) => {
      const grad = defs.append("linearGradient")
        .attr("id", `hh-${ key(d.source, d.target) }`);

      grad.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", hubColor(d.source.hubDegree));

      grad.append("stop")
        .attr("offset", "100%")
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
      .attr("stroke", d =>
        `url(#hh-${ key(d.source, d.target) })`
      )
      .attr("stroke-opacity", (d) => hhOpacity(d.weight))
      .attr("stroke-width", (d) => hhWidth(d.weight));

    // Neighbour nodes -------------------------------------------------------
    const nbNodes = nodes.filter(n => n.kind === "neighbour");
    const nbGroup = container.append("g").attr("class", "nb-nodes")
      .selectAll<SVGGElement, ContextNode>("g")
      .data(nbNodes, d => d.id).join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setControls('selectedNode', (prev: any) => prev === d.id ? null : d.id))
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
      .on("click", (_, d) => setControls('selectedNode', prev => prev === d.id ? null : d.id))
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
              ? Math.max(40, (260 - (d as HubHubEdge).weight * 180) * controls.hubSpread)
              : 60
          )
          .strength(d => d.kind === "hub-hub" ? 0.55 : 0.8)
      )
      .force("charge", d3.forceManyBody()
        // .strength((d) => d.kind === "hub" ? -280 : -40)
        .strength(d => (d as any).kind === "hub" ? -280 * controls.hubSpread : -40)
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

  const totalEventsForConcept = createMemo(() => {
    const cd = tier2Data[controls.concept];
    return cd?.n_events ?? 0;
  });

  return (
    <article class="svg-cg-layout no-padding no-margin">

      <ControlsHeader
        totalEvents={totalEventsForConcept}
        includeHubSpread={true}
        concepts={concepts}
        MAX_TOP_N={MAX_TOP_N}
        yearFiltered={yearFiltered}
        yearBounds={yearBounds}
      />

      <div class="cg-main background">
        <svg ref={svgRef!} class="cg-svg surface-container-lowest" />

        <Show when={controls.selectedNode}>
          <aside class="cg-aside surface-container-high padding border">

            <div class="cg-header-row">
              <h2>{controls.selectedNode}</h2>
              <button class="link border" onClick={() => setControls('selectedNode', null)}>✕</button>
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
                      <div>Hub connections: {graphData().nodes.find(n => n.id === controls.selectedNode)?.hubDegree ?? 0}</div>
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
                {controls.viewMode === "aggregated" ? "Hub contexts" : "Event contexts"}
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
          <Show when={controls.viewMode === "aggregated"}>
            <span class="cg-legend-hub" />hubs ({graphData().nodes.filter(n => n.kind === "hub").length})
          </Show>
          <Show when={controls.viewMode === "events"}>
            <span class="cg-legend-event" />events ({graphData().nodes.filter(n => n.kind === "event").length})
          </Show>
          <span class="cg-legend-nb" />neighbours ({graphData().nodes.filter(n => n.kind === "neighbour").length})
          <Show when={controls.viewMode === "aggregated"}>
            {" • "}{graphData().hubHubEdges.length} similarity edges
          </Show>
          {" • "}{graphData().hubNbEdges.length} spokes
          {" • "}{yearFiltered().length} events
          <Show when={controls.fromYear !== yearBounds()[0] || controls.toYear !== yearBounds()[1]}>
            {" • "}{controls.fromYear}–{controls.toYear}
          </Show>
        </span>
      </footer>

    </article>
  );
};

export default ContextGraph5;
