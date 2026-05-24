/**
 * ConceptGraph.tsx
 *
 * Architecture
 * ------------
 *   ConceptData (raw API)
 *       ↓  filterByYearRange()      — optional, driven by year slider
 *       ↓  aggregateConcept()       — O(n²), memoised per concept+year window
 *   AggregatedConcept               — carries full provenance incl. pub_year
 *       ↓  buildGraph()             — cheap, reruns on filter change
 *   GraphData                       — D3-facing, no provenance
 *       ↓  render()
 *   SVG
 *
 * Year filtering
 * --------------
 * scanYearRange() derives slider bounds from the concept's events at load
 * time. The year range signals drive filterByYearRange() before aggregation,
 * so the graph reflects only events from the selected window.
 *
 * Drill-down
 * ----------
 * Node click opens a panel reading AggregatedConcept.byToken[id].docs
 * directly — a Map<doc_id, pub_year> so years display without re-scanning.
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

import {
  aggregateConcept,
  buildGraph,
  filterByYearRange,
  scanYearRange,
  EMPTY_GRAPH,
} from "./ConceptGraph.data";

import type {
  Tier2Data,
  AggregatedConcept,
  GraphNode,
  GraphEdge,
  GraphData,
} from "./ConceptGraph.types";

interface Props {
  data: Tier2Data;
}

const EEBO_CONFIG_SLICES_MIN = 1625;
const EEBO_CONFIG_SLICES_MAX = 1665;

const showDocument = (docId: string) => {
  const url = `/api/${ docId }`;
  window.open(url, "_blank", "noopener,noreferrer");
}

const ConceptGraph: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxNodes, setMaxNodes] = createSignal(50);
  const [minEdge, setMinEdge] = createSignal(3);
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);

  // Year range
  // Bounds are derived from the current concept's events.
  // fromYear/toYear signals are reset whenever the concept changes.

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = props.data[concept()];
    if (!cd) return [EEBO_CONFIG_SLICES_MIN, EEBO_CONFIG_SLICES_MAX];
    const [min, max] = scanYearRange(cd);
    return [min ?? EEBO_CONFIG_SLICES_MIN, max ?? EEBO_CONFIG_SLICES_MAX];
  });

  const [fromYear, setFromYear] = createSignal<number>(yearBounds()[0]);
  const [toYear, setToYear] = createSignal<number>(yearBounds()[1]);

  // Reset year signals when concept (and therefore bounds) changes
  createEffect(() => {
    const [min, max] = yearBounds();
    setFromYear(min);
    setToYear(max);
  });

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    // If the slider covers the full range, skip filtering entirely
    if (fromYear() <= min && toYear() >= max) return cd.events;
    return filterByYearRange(cd.events, fromYear(), toYear());
  });

  // Stage 1: aggregation
  // Reruns when concept or year window changes.

  const aggregated = createMemo<AggregatedConcept | null>(() => {
    const cd = props.data[concept()];
    if (!cd) return null;
    return aggregateConcept(cd, yearFiltered());
  });

  // Stage 2: graph construction
  // Reruns only when filters or aggregated data change.

  const graphData = createMemo<GraphData>(() => {
    const agg = aggregated();
    if (!agg) return EMPTY_GRAPH;
    return buildGraph(agg, minEdge(), maxNodes());
  });

  // Drill-down

  const selectedStats = createMemo(() => {
    const node = selectedNode();
    const agg = aggregated();
    if (!node || !agg) return null;
    return agg.byToken.get(node) ?? null;
  });

  // Sorted list of [doc_id, pub_year] for the selected node
  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const stats = selectedStats();
    if (!stats) return [];
    return [...stats.docs.entries()].sort((a, b) => {
      // Sort by year ascending, undated docs last
      const ya = a[1] ?? Infinity;
      const yb = b[1] ?? Infinity;
      return ya - yb;
    });
  });

  // D3

  let svgRef!: SVGSVGElement;
  let simulationRef: d3.Simulation<GraphNode, GraphEdge> | null = null;

  function render() {
    const { nodes, edges, maxWeight, maxDegree } = graphData();
    const svg = d3.select(svgRef);
    const W = svgRef.clientWidth;
    const H = svgRef.clientHeight;

    svg.selectAll("*").remove();

    if (nodes.length === 0) {
      svg
        .append("text")
        .attr("x", W / 2)
        .attr("y", H / 2)
        .attr("text-anchor", "middle")
        .attr("fill", "#933")
        .attr("font-family", "monospace")
        .text("No graph — try reducing min edge weight or widening the year range");
      return;
    }

    const nodeRadius = d3.scaleSqrt().domain([0, maxDegree]).range([4, 18]);
    const edgeOpacity = d3.scaleLinear().domain([0, maxWeight]).range([0.5, 1]);
    const edgeWidth = d3.scaleLinear().domain([0, maxWeight]).range([2, 10]);
    const nodeColor = d3.scaleLinear<string>().domain([0, maxDegree]).range(["#5a87ba66", "#e9f3fcdd"]);

    const container = svg.append("g").attr("class", "zoom-container");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => container.attr("transform", event.transform))
    );

    const defs = container.append("defs");

    edges.forEach((d, i) => {
      const grad = defs
        .append("linearGradient")
        .attr("id", `eg-${ i }`)
        .attr("gradientUnits", "userSpaceOnUse");
      grad.append("stop").attr("offset", "0%").attr("stop-color", nodeColor(d.source.degree));
      grad.append("stop").attr("offset", "100%").attr("stop-color", nodeColor(d.target.degree));
    });

    const edgeSelection = container
      .append("g").attr("class", "edges")
      .selectAll<SVGLineElement, GraphEdge>("line")
      .data(edges).join("line")
      .attr("stroke", (_, i) => `url(#eg-${ i })`)
      .attr("stroke-opacity", (d) => edgeOpacity(d.weight))
      .attr("stroke-width", (d) => edgeWidth(d.weight));

    const nodeGroup = container
      .append("g").attr("class", "nodes")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode((prev) => prev === d.id ? null : d.id))
      .call(
        d3.drag<SVGGElement, GraphNode>()
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

    nodeGroup.append("text")
      .text((d) => d.id)
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
        const stats = aggregated()?.byToken.get(d.id);
        tooltip
          .html(
            `<aside>` +
            `<h6 class="bottom-padding">${ d.id }</h6>` +
            `Connections: ${ d.degree }<br/>` +
            `Documents: ${ stats?.docs.size ?? "—" }<br/>` +
            `Appearances: ${ stats?.totalAppearances ?? "—" }` +
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

    simulationRef = d3.forceSimulation<GraphNode>(nodes)
      .force("link",
        d3.forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((d) => Math.max(100, 200 - d.weight * 3))
          .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("collision", d3.forceCollide<GraphNode>().radius((d) => nodeRadius(d.degree) + 6))
      .on("tick", () => {
        edgeSelection
          .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
          .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
          .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
          .attr("y2", (d) => (d.target as GraphNode).y ?? 0);

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

  // UI

  return (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%", width: "100%" }}>

      <header
        class="fill responsive surface-container-low"
        style={{ display: "flex", gap: "2rem", "align-items": "center", "flex-shrink": "0", "flex-wrap": "wrap" }}
      >
        {/* Concept */}
        <div>
          <label>Concept </label>
          <select value={concept()} onChange={(e) => setConcept(e.currentTarget.value)}>
            <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
          </select>
        </div>

        {/* Max nodes */}
        <div>
          <label>Max nodes </label>
          <select value={maxNodes()} onChange={(e) => setMaxNodes(Number(e.currentTarget.value))}>
            <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
          </select>
        </div>

        {/* Min edge */}
        <div>
          <label>Min edge {minEdge()} </label>
          <input type="range" min={1} max={10} step={1} value={minEdge()}
            onInput={(e) => setMinEdge(Number(e.currentTarget.value))} />
        </div>

        {/* Year range — only shown when year data is present */}
        <Show when={yearBounds()[0] !== yearBounds()[1]}>
          <div style={{ display: "flex", gap: "0.75rem", "align-items": "center" }}>
            <label>From</label>
            <input class="border" type="number" min={yearBounds()[0]} max={toYear()} step={1}
              value={fromYear()}
              onInput={(e) => {
                const v = Number(e.currentTarget.value);
                if (!isNaN(v)) setFromYear(Math.min(v, toYear()));
              }}
            />
            <label>To</label>
            <input class="border" type="number" min={fromYear()} max={yearBounds()[1]} step={1}
              value={toYear()}
              onInput={(e) => {
                const v = Number(e.currentTarget.value);
                if (!isNaN(v)) setToYear(Math.max(v, fromYear()));
              }}
            />
            <span class="small-text">
              {yearFiltered().length} / {props.data[concept()]?.n_events ?? 0} occurrences
            </span>
          </div>
        </Show>
      </header>

      {/* Main area */}
      <div style={{ display: "flex", flex: "1", overflow: "hidden" }} class="background">

        <svg ref={svgRef!} style={{ flex: "1", display: "block" }} class="surface-container-lowest" />

        {/* Drill-down panel */}
        <Show when={selectedNode()}>
          <aside class="surface-container-high padding border"
            style={{ width: "20rem", "flex-shrink": "0", "overflow-y": "auto" }}
          >
            <div style={{ display: "flex", "justify-content": "space-between", "align-items": "center" }}>
              <h2>{selectedNode()}</h2>
              <button class="link border" onClick={() => setSelectedNode(null)}>✕</button>
            </div>

            <Show when={selectedStats()}>
              {(stats) => (
                <div class="bottom-padding">
                  <div>Connections: {graphData().nodes.find(n => n.id === selectedNode())?.degree ?? "—"}</div>
                  <div>Appearances: {stats().totalAppearances}</div>
                  <div>Documents: {stats().docs.size}</div>
                </div>
              )}
            </Show>

            <h3 class="bottom-padding">Sources</h3>

            <Show
              when={selectedDocs().length > 0}
              fallback={<div class="error">No documents found</div>}
            >
              <For each={selectedDocs()}>
                {([docId, pubYear]) => (
                  <button class="chip small-margin" onClick={() => showDocument(docId)}>
                    <span>{docId}</span>
                    <Show when={pubYear !== undefined}>
                      <span class="small-text"> {pubYear} </span>
                    </Show>
                  </button>
                )}
              </For>
            </Show>
          </aside>
        </Show>

      </div>

      <footer class="fixed responsive small-padding surface-container-low" style={{ "flex-shrink": "0" }}>
        {graphData().nodes.length} nodes
        {" • "}
        {graphData().edges.length} edges
        {" • "}
        {yearFiltered().length} occurrences
        <Show when={fromYear() !== yearBounds()[0] || toYear() !== yearBounds()[1]}>
          {" • "}
          <span style={{ opacity: "0.6" }}>{fromYear()}–{toYear()}</span>
        </Show>
        <Show when={selectedNode()}>
          {" • "}
          <span style={{ opacity: "0.6" }}>{selectedNode()} — {selectedDocs().length} docs</span>
        </Show>
      </footer>

    </div>
  );
};

export default ConceptGraph;
