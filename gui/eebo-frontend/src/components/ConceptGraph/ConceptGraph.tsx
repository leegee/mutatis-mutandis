/**
 * ConceptGraph.tsx
 *
 * Interactive co-occurrence graph for Tier2 concept neighbour data.
 * Receives pre-built concept data as a prop; all graph state is local.
 *
 * Architecture
 * ------------
 * Data flows through three stages:
 *
 *   ConceptData (raw API)
 *       ↓  aggregateConcept()       — O(n²), memoised per concept
 *   AggregatedConcept               — carries full provenance
 *       ↓  buildGraph()             — cheap, reruns on filter change
 *   GraphData                       — D3-facing, no provenance
 *       ↓  render()
 *   SVG
 *
 * This separation means:
 *   - Filter changes (min edge, max nodes) only rerun buildGraph
 *   - Document drill-down reads AggregatedConcept directly — no
 *     provenance needs to live on GraphNode
 *   - Temporal split view (future) calls aggregateConcept with
 *     year-filtered events then buildGraph twice — no structural change
 *
 * See conceptGraph.data.ts for aggregateConcept, buildGraph, filterByYearRange.
 * See conceptGraph.types.ts for all shared types.
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


const ConceptGraph: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxNodes, setMaxNodes] = createSignal(50);
  const [minEdge, setMinEdge] = createSignal(3);

  // Drill-down state: which node is selected, if any
  const [selectedNode, setSelectedNode] = createSignal<string | null>(null);

  let svgRef!: SVGSVGElement;
  let simulationRef: d3.Simulation<GraphNode, GraphEdge> | null = null;

  // Stage 1: aggregation
  // O(n²) — reruns only when concept changes.
  // Carries full provenance (doc_ids, co-occurrence counts).

  const aggregated = createMemo<AggregatedConcept | null>(() => {
    const cd = props.data[concept()];
    if (!cd) return null;
    return aggregateConcept(cd);
  });

  // Stage 2: graph construction
  // Cheap — reruns when filters or aggregated data change.
  // Produces D3-ready nodes and edges; no provenance.

  const graphData = createMemo<GraphData>(() => {
    const agg = aggregated();
    if (!agg) return EMPTY_GRAPH;
    return buildGraph(agg, minEdge(), maxNodes());
  });

  // Drill-down: doc_ids for selected node
  // Reads directly from AggregatedConcept — no graph layer involvement.

  const selectedDocs = createMemo<string[]>(() => {
    const node = selectedNode();
    const agg = aggregated();
    if (!node || !agg) return [];
    return [...(agg.byToken.get(node)?.docs ?? [])].sort();
  });

  const selectedStats = createMemo(() => {
    const node = selectedNode();
    const agg = aggregated();
    if (!node || !agg) return null;
    return agg.byToken.get(node) ?? null;
  });

  // D3 rendering

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
        .text("No graph — try reducing min edge weight");
      return;
    }

    const nodeRadius = d3.scaleSqrt().domain([0, maxDegree]).range([4, 18]);

    const edgeOpacity = d3
      .scaleLinear()
      .domain([0, maxWeight])
      .range([0.5, 1]);

    const edgeWidth = d3
      .scaleLinear()
      .domain([0, maxWeight])
      .range([2, 10]);

    const nodeColor = d3
      .scaleLinear<string>()
      .domain([0, maxDegree])
      .range(["#5a87ba", "#e9f3fc"]);

    const container = svg.append("g").attr("class", "zoom-container");

    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
          container.attr("transform", event.transform);
        })
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
      .append("g")
      .attr("class", "edges")
      .selectAll<SVGLineElement, GraphEdge>("line")
      .data(edges)
      .join("line")
      .attr("stroke", (_, i) => `url(#eg-${ i })`)
      .attr("stroke-opacity", (d) => edgeOpacity(d.weight))
      .attr("stroke-width", (d) => edgeWidth(d.weight));

    const nodeGroup = container
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => {
        // Toggle selection — clicking the same node again clears it
        setSelectedNode((prev) => (prev === d.id ? null : d.id));
      })
      .call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on("start", (event, d) => {
            if (!event.active) simulationRef?.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) simulationRef?.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    nodeGroup
      .append("circle")
      .attr("r", (d) => nodeRadius(d.degree))
      .attr("fill", (d) => nodeColor(d.degree))
      .attr("stroke", "rgba(200,230,255,0.25)")
      .attr("stroke-width", 1);

    nodeGroup
      .append("text")
      .text((d) => d.id)
      .attr("dy", (d) => -nodeRadius(d.degree) - 3)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("font-family", "'IBM Plex Mono', 'Courier New', monospace")
      .attr("fill", "rgba(210,235,255,0.85)")
      .attr("pointer-events", "none");

    // Tooltip
    const tooltip = d3
      .select("body")
      .selectAll<HTMLDivElement, unknown>(".cg-tooltip")
      .data([null])
      .join("div")
      .attr("class", "cg-tooltip")
      .style("position", "fixed")
      .style("pointer-events", "none")
      .style("background", "rgba(8,16,28,0.93)")
      .style("border", "1px solid #2a4a6a")
      .style("color", "#c8e6ff")
      .style("font-family", "'IBM Plex Mono', monospace")
      .style("font-size", "11px")
      .style("padding", "6px 10px")
      .style("border-radius", "3px")
      .style("opacity", "0")
      .style("transition", "opacity 0.15s");

    nodeGroup
      .on("mouseenter", (event, d) => {
        const stats = aggregated()?.byToken.get(d.id);
        tooltip
          .html(
            `<strong>${ d.id }</strong><br/>` +
            `connections: ${ d.degree }<br/>` +
            `documents: ${ stats?.docs.size ?? "—" }<br/>` +
            `appearances: ${ stats?.totalAppearances ?? "—" }`
          )
          .style("opacity", "1")
          .style("left", event.clientX + 14 + "px")
          .style("top", event.clientY - 10 + "px");
      })
      .on("mousemove", (event) => {
        tooltip
          .style("left", event.clientX + 14 + "px")
          .style("top", event.clientY - 10 + "px");
      })
      .on("mouseleave", () => {
        tooltip.style("opacity", "0");
      });

    if (simulationRef) simulationRef.stop();

    simulationRef = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          .distance((d) => Math.max(100, 200 - d.weight * 3))
          .strength(0.6)
      )
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force(
        "collision",
        d3.forceCollide<GraphNode>().radius((d) => nodeRadius(d.degree) + 6)
      )
      .on("tick", () => {
        edgeSelection
          .attr("x1", (d) => (d.source as GraphNode).x ?? 0)
          .attr("y1", (d) => (d.source as GraphNode).y ?? 0)
          .attr("x2", (d) => (d.target as GraphNode).x ?? 0)
          .attr("y2", (d) => (d.target as GraphNode).y ?? 0);

        edges.forEach((d, i) => {
          defs
            .select(`#eg-${ i }`)
            .attr("x1", d.source.x ?? 0)
            .attr("y1", d.source.y ?? 0)
            .attr("x2", d.target.x ?? 0)
            .attr("y2", d.target.y ?? 0);
        });

        nodeGroup.attr(
          "transform",
          (d) => `translate(${ d.x ?? 0 },${ d.y ?? 0 })`
        );
      });
  }

  createEffect(() => {
    graphData();
    if (svgRef) render();
  });

  onCleanup(() => {
    simulationRef?.stop();
    d3.select("body").selectAll(".cg-tooltip").remove();
  });


  return (
    <div
      style={{
        display: "flex",
        "flex-direction": "column",
        height: "100%",
        width: "100%",
      }}
    >
      <header
        class="padding border bottom-border"
        style={{
          display: "flex",
          gap: "2rem",
          "align-items": "center",
          "flex-shrink": "0",
        }}
      >
        <h1 style="font-size: 1rem">Concept Graph</h1>

        <div>
          <label>Concept </label>
          <select value={concept()} onChange={(e) => setConcept(e.currentTarget.value)}>
            <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
          </select>
        </div>

        <div>
          <label>Max nodes </label>
          <select value={maxNodes()} onChange={(e) => setMaxNodes(Number(e.currentTarget.value))}>
            <For each={[10, 20, 50, 100]}>{(n) => <option value={n}>{n}</option>}</For>
          </select>
        </div>

        <div>
          <label>Min edge {minEdge()} </label>
          <input
            type="range"
            min={1} max={10} step={1}
            value={minEdge()}
            onInput={(e) => setMinEdge(Number(e.currentTarget.value))}
          />
        </div>
      </header>

      {/* Main area: graph + optional drill-down panel */}
      <div style={{ display: "flex", flex: "1", overflow: "hidden" }}>

        <svg ref={svgRef!} style={{ flex: "1", display: "block" }} />

        {/* Drill-down panel — shown when a node is selected */}
        <Show when={selectedNode()}>
          <aside class="surface padding"
            style={{
              width: "20rem",
              "flex-shrink": "0",
              "overflow-y": "auto",
            }}
          >
            <div style={{ display: "flex", "justify-content": "space-between", "align-items": "center" }}>
              <strong>
                {selectedNode()}
              </strong>
              <button class="link border" onClick={() => setSelectedNode(null)} >
                ✕
              </button>
            </div>

            <Show when={selectedStats()}>
              {(stats) => (
                <div class="bottom-padding">
                  <div>connections: {graphData().nodes.find(n => n.id === selectedNode())?.degree ?? "—"}</div>
                  <div>appearances: {stats().totalAppearances}</div>
                  <div>documents: {stats().docs.size}</div>
                </div>
              )}
            </Show>

            <div class="bottom-padding">
              Source documents
            </div>

            <Show
              when={selectedDocs().length > 0}
              fallback={<div class="error">No documents found</div>}
            >
              <For each={selectedDocs()}>
                {(docId) => (
                  <div>
                    {docId}
                  </div>
                )}
              </For>
            </Show>
          </aside>
        </Show>

      </div>

      <footer
        class="fixed responsive small-padding border"
        style={{ "flex-shrink": "0" }}
      >
        {graphData().nodes.length} nodes · {graphData().edges.length} edges
        {" · "}
        {props.data[concept()]?.n_events ?? 0} events
        <Show when={selectedNode()}>
          {" · "}
          <span style={{ color: "#4a7a9b" }}>
            {selectedNode()} — {selectedDocs().length} docs
          </span>
        </Show>
      </footer>
    </div>
  );
};

export default ConceptGraph;