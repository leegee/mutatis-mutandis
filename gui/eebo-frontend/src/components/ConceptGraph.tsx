/**
 * ConceptGraph.tsx
 *
 * Interactive co-occurrence graph for Tier2 concept neighbour data.
 * Receives pre-built concept data as a prop; all graph state is local.
 *
 * Features
 * --------
 * - D3 force simulation with zoom + pan (replaces Plotly)
 * - Concept selector dropdown
 * - Max-nodes and min-edge-weight controls
 * - Node degree encoded as size + brightness
 * - Edge weight encoded as opacity + thickness
 * - Per-edge linearGradient from source to target node colour
 *
 * Usage
 * -----
 *   import ConceptGraph from "./ConceptGraph";
 *   import data from "./tier2_concept_neighbours.json";
 *
 *   <ConceptGraph data={data} />
 *
 * Data shape (subset of Tier2 output)
 * ------------------------------------
 *   Record<string, {
 *     n_events: number;
 *     events: Array<{
 *       neighbours: Array<{ token: string; score: number; }>
 *     }>
 *   }>
 */

import {
  createSignal,
  createMemo,
  createEffect,
  onCleanup,
  For,
  type Component,
} from "solid-js";

import * as d3 from "d3";
import ConceptGraphGuide from "./ConceptGraphGuide";
import { Transition } from "solid-transition-group";

export interface Tier2Data {
  [concept: string]: ConceptData;
}

interface Neighbour {
  token: string;
  score: number;
  event_id?: number;
  doc_id?: string;
  window_id?: number;
}

interface ConceptEvent {
  event_id?: number;
  token?: string;
  neighbours: Neighbour[];
}

interface ConceptData {
  n_events: number;
  events: ConceptEvent[];
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  degree: number;
}

interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  source: GraphNode;
  target: GraphNode;
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  maxWeight: number;
  maxDegree: number;
}

interface Props {
  data: Tier2Data;
}

// Graph construction mirrors Python build_graph / subset_graph
function buildGraph(
  conceptData: ConceptData,
  minEdgeWeight: number,
  maxNodes: number
): GraphData {
  const edgeWeights = new Map<string, number>();

  for (const event of conceptData.events) {
    const tokens = event.neighbours.map((n) => n.token);

    for (let i = 0; i < tokens.length; i++) {
      for (let j = i + 1; j < tokens.length; j++) {
        const key = [tokens[i], tokens[j]].sort().join("\x00");
        edgeWeights.set(key, (edgeWeights.get(key) ?? 0) + 1);
      }
    }
  }

  // Filter by min edge weight
  const filteredEdges: Array<[string, string, number]> = [];
  for (const [key, w] of edgeWeights) {
    if (w >= minEdgeWeight) {
      const [a, b] = key.split("\x00");
      filteredEdges.push([a, b, w]);
    }
  }

  // Compute degree for each node in filtered graph
  const degreeMap = new Map<string, number>();

  for (const [a, b] of filteredEdges) {
    degreeMap.set(a, (degreeMap.get(a) ?? 0) + 1);
    degreeMap.set(b, (degreeMap.get(b) ?? 0) + 1);
  }

  // Select top-maxNodes nodes by degree
  const sortedNodes = [...degreeMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxNodes);

  const keepSet = new Set(sortedNodes.map(([id]) => id));

  const nodes: GraphNode[] = sortedNodes.map(([id, degree]) => ({
    id,
    degree,
  }));

  const edges: GraphEdge[] = filteredEdges
    .filter(([a, b]) => keepSet.has(a) && keepSet.has(b))
    .map(([a, b, weight]) => ({
      source: nodes.find((n) => n.id === a)!,
      target: nodes.find((n) => n.id === b)!,
      weight,
    }));

  const maxWeight = Math.max(1, ...edges.map((e) => e.weight));
  const maxDegree = Math.max(1, ...nodes.map((n) => n.degree));

  return { nodes, edges, maxWeight, maxDegree };
}


const ConceptGraph: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);
  const [openHelp, setOpenHelp] = createSignal(false);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxNodes, setMaxNodes] = createSignal(50);
  const [minEdge, setMinEdge] = createSignal(3);

  let svgRef!: SVGSVGElement;
  let simulationRef: d3.Simulation<GraphNode, GraphEdge> | null = null;

  // derived graph data
  const graphData = createMemo<GraphData>(() => {
    const c = concept();
    const cd = props.data[c];
    if (!cd) return { nodes: [], edges: [], maxWeight: 1, maxDegree: 1 };
    return buildGraph(cd, minEdge(), maxNodes());
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

    // Scales
    const nodeRadius = d3
      .scaleSqrt()
      .domain([0, maxDegree])
      .range([4, 18]);

    const edgeOpacity = d3
      .scaleLinear()
      .domain([0, maxWeight])
      .range([0.5, 1]);

    const edgeWidth = d3
      .scaleLinear()
      .domain([0, maxWeight])
      .range([1, 6]);

    const nodeColor = d3
      .scaleLinear<string>()
      .domain([0, maxDegree])
      .range(["#5a87ba", "#e9f3fc"]);

    // Zoom container
    const container = svg.append("g").attr("class", "zoom-container");
    svg.call(
      d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) => {
          container.attr("transform", event.transform);
        })
    );

    // Per-edge linearGradient from source node colour to target node colour.
    // gradientUnits="userSpaceOnUse" means x1/y1/x2/y2 are in the same
    // coordinate space as the nodes, so the gradient follows the line as
    // nodes move during simulation. Updated on every tick below.
    const defs = container.append("defs");

    edges.forEach((d, i) => {
      const grad = defs
        .append("linearGradient")
        .attr("id", `eg-${ i }`)
        .attr("gradientUnits", "userSpaceOnUse");

      grad
        .append("stop")
        .attr("offset", "0%")
        .attr("stop-color", nodeColor(d.source.degree));

      grad
        .append("stop")
        .attr("offset", "100%")
        .attr("stop-color", nodeColor(d.target.degree));
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

    // Nodes
    const nodeGroup = container
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
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
        tooltip
          .html(`<strong>${ d.id }</strong><br/>degree: ${ d.degree }`)
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

    // Force simulation
    if (simulationRef) simulationRef.stop();

    simulationRef = d3
      .forceSimulation<GraphNode>(nodes)
      .force(
        "link",
        d3
          .forceLink<GraphNode, GraphEdge>(edges)
          .id((d) => d.id)
          // .distance((d) => 80 - d.weight * 2)
          .distance((d) => Math.max(50, 120 - d.weight * 3))
          .strength(0.6) // was 0.4
      )
      // .force("charge", d3.forceManyBody().strength(-180))
      .force("charge", d3.forceManyBody().strength(-80))
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

        // Keep gradient coordinate space aligned with node positions
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

  // createEffect is the correct SolidJS primitive for D3 side effects
  // driven by reactive state. It runs after mount (svgRef is available)
  // and re-runs whenever graphData() changes. onMount + createMemo was
  // incorrect: createMemo is for deriving values, not side effects, and
  // the combination produced a double render on load.
  createEffect(() => {
    graphData(); // subscribe to memo
    if (svgRef) render();
  });

  onCleanup(() => {
    simulationRef?.stop();
    d3.select("body").selectAll(".cg-tooltip").remove();
  });

  // UI

  return (
    <>

      <div
        style={{
          display: "flex",
          "flex-direction": "column",
          height: "100%",
          width: "100%",
          background: "#080e18",
          color: "#c8e6ff",
          "font-family": "'IBM Plex Mono', 'Courier New', monospace",
        }}
      >
        <header
          style={{
            display: "flex",
            "align-items": "center",
            gap: "2rem",
            padding: "0.75rem 1.25rem",
            "border-bottom": "1px solid #1a2e42",
            "flex-shrink": "0",
          }}
        >
          <span
            style={{
              "font-size": "11px",
              "letter-spacing": "0.15em",
              "text-transform": "uppercase",
              color: "#4a7a9b",
            }}
          >
            Concept Graph
          </span>

          {/* Concept selector */}
          <div style={{ display: "flex", "align-items": "center", gap: "0.5rem" }}>
            <label style={{ "font-size": "10px", color: "#4a7a9b" }}>
              concept
            </label>
            <select
              value={concept()}
              onChange={(e) => setConcept(e.currentTarget.value)}
              style={{
                background: "#0d1a28",
                border: "1px solid #1e3a52",
                color: "#c8e6ff",
                "font-family": "inherit",
                "font-size": "11px",
                padding: "3px 6px",
                cursor: "pointer",
              }}
            >
              <For each={concepts}>
                {(c) => <option value={c}>{c}</option>}
              </For>
            </select>
          </div>

          {/* Max nodes */}
          <div style={{ display: "flex", "align-items": "center", gap: "0.5rem" }}>
            <label style={{ "font-size": "10px", color: "#4a7a9b" }}>
              max nodes
            </label>
            <select
              value={maxNodes()}
              onChange={(e) => setMaxNodes(Number(e.currentTarget.value))}
              style={{
                background: "#0d1a28",
                border: "1px solid #1e3a52",
                color: "#c8e6ff",
                "font-family": "inherit",
                "font-size": "11px",
                padding: "3px 6px",
                cursor: "pointer",
              }}
            >
              <For each={[10, 20, 50, 100]}>
                {(n) => <option value={n}>{n}</option>}
              </For>
            </select>
          </div>

          {/* Min edge weight */}
          <div
            style={{
              display: "flex",
              "align-items": "center",
              gap: "0.5rem",
              flex: "1",
              "max-width": "220px",
            }}
          >
            <label
              style={{
                "font-size": "10px",
                color: "#4a7a9b",
                "white-space": "nowrap",
              }}
            >
              min edge {minEdge()}
            </label>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={minEdge()}
              onInput={(e) => setMinEdge(Number(e.currentTarget.value))}
              style={{ flex: "1", "accent-color": "#4a7a9b" }}
            />
          </div>

          {/* Stats */}
          <div
            style={{
              "margin-left": "auto",
              "font-size": "10px",
              color: "#2a5a7a",
            }}
          >
            {graphData().nodes.length} nodes · {graphData().edges.length} edges
            {" · "}
            {props.data[concept()]?.n_events ?? 0} events
          </div>

          <button class="border small" onClick={() => setOpenHelp(v => !v)}>
            <i>help</i>
          </button>

        </header>

        {/* Graph canvas */}
        <svg
          ref={svgRef!}
          style={{ flex: "1", display: "block", width: "100%" }}
        />

        {/* Footer hint */}
        <footer
          style={{
            padding: "0.35rem 1.25rem",
            "font-size": "9px",
            color: "#1e3a52",
            "border-top": "1px solid #0e1e2e",
            "flex-shrink": "0",
          }}
        >
          scroll to zoom · drag to pan · drag nodes to reposition
        </footer>
      </div>


      <Transition name="slide-fade">
        {openHelp() && (
          <article class="helpContainer">
            <ConceptGraphGuide />
          </article>
        )}
      </Transition>

    </>
  );
};

export default ConceptGraph;