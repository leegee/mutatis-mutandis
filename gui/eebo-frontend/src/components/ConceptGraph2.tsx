/**
 * ConceptGraph.tsx
 *
 * Self-contained experimental event-overlap graph.
 *
 * Nodes:
 *   individual concept usage events
 *
 * Edges:
 *   overlap between FAISS KNN sets
 *
 * Goal:
 *   surface contextual structure rather than aggregate token frequency.
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

/* -------------------------------------------------------------------------- */
/* Types */
/* -------------------------------------------------------------------------- */

interface Neighbour {
  token: string;
  score: number;
  pub_year?: number;
}

interface ConceptEvent {
  id: string;
  token: string;
  doc_id?: string;
  pub_year?: number;
  neighbours: Neighbour[];
}

interface ConceptData {
  events: ConceptEvent[];
  n_events?: number;
}

type Tier2Data = Record<string, ConceptData>;

interface EventNode extends d3.SimulationNodeDatum {
  id: string;
  token: string;
  doc_id?: string;
  pub_year?: number;
  neighbours: Neighbour[];
  degree: number;
}

interface EventEdge extends d3.SimulationLinkDatum<EventNode> {
  source: EventNode;
  target: EventNode;
  overlap: number;
  weight: number;
  percentile: number;
}

interface GraphData {
  nodes: EventNode[];
  edges: EventEdge[];
  maxWeight: number;
  maxDegree: number;
}

/* -------------------------------------------------------------------------- */
/* Constants */
/* -------------------------------------------------------------------------- */

const CORPUS_START_YEAR = 1470;
const CORPUS_END_YEAR = 1700;

const EMPTY_GRAPH: GraphData = {
  nodes: [],
  edges: [],
  maxWeight: 1,
  maxDegree: 1,
};

/* -------------------------------------------------------------------------- */
/* Utilities */
/* -------------------------------------------------------------------------- */

function scanYearRange(cd: ConceptData): [number, number] {
  let min = Infinity;
  let max = -Infinity;

  for (const ev of cd.events) {
    if (ev.pub_year === undefined) continue;
    min = Math.min(min, ev.pub_year);
    max = Math.max(max, ev.pub_year);
  }

  if (!isFinite(min) || !isFinite(max)) {
    return [CORPUS_START_YEAR, CORPUS_END_YEAR];
  }

  return [min, max];
}

function filterByYearRange(
  events: ConceptEvent[],
  from: number,
  to: number
): ConceptEvent[] {
  return events.filter((e) => {
    if (e.pub_year === undefined) return false;
    return e.pub_year >= from && e.pub_year <= to;
  });
}

function percentile(values: number[], x: number): number {
  if (values.length === 0) return 0;

  let below = 0;

  for (const v of values) {
    if (v <= x) below++;
  }

  return below / values.length;
}

/* -------------------------------------------------------------------------- */
/* Event overlap graph */
/* -------------------------------------------------------------------------- */

function buildEventGraph(
  events: ConceptEvent[],
  minPercentile: number,
  maxNodes: number
): GraphData {
  if (events.length === 0) return EMPTY_GRAPH;

  const sliced = events.slice(0, maxNodes);

  const nodes: EventNode[] = sliced.map((e) => ({
    id: e.id,
    token: e.token,
    doc_id: e.doc_id,
    pub_year: e.pub_year,
    neighbours: e.neighbours ?? [],
    degree: 0,
  }));

  const neighbourSets = new Map<string, Set<string>>();

  for (const n of nodes) {
    neighbourSets.set(
      n.id,
      new Set(n.neighbours.map((x) => x.token.toLowerCase()))
    );
  }

  const raw: {
    source: EventNode;
    target: EventNode;
    overlap: number;
  }[] = [];

  const overlaps: number[] = [];

  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i];
      const b = nodes[j];

      const A = neighbourSets.get(a.id)!;
      const B = neighbourSets.get(b.id)!;

      let overlap = 0;

      for (const tok of A) {
        if (B.has(tok)) overlap++;
      }

      overlaps.push(overlap);

      raw.push({
        source: a,
        target: b,
        overlap,
      });
    }
  }

  const edges: EventEdge[] = [];

  for (const r of raw) {
    const p = percentile(overlaps, r.overlap);

    if (p < minPercentile / 10) continue;

    edges.push({
      source: r.source,
      target: r.target,
      overlap: r.overlap,
      weight: r.overlap,
      percentile: p,
    });

    r.source.degree++;
    r.target.degree++;
  }

  const connected = nodes.filter((n) => n.degree > 0);

  return {
    nodes: connected,
    edges,
    maxWeight: Math.max(1, ...edges.map((e) => e.weight)),
    maxDegree: Math.max(1, ...connected.map((n) => n.degree)),
  };
}

/* -------------------------------------------------------------------------- */
/* Component */
/* -------------------------------------------------------------------------- */

interface Props {
  data: Tier2Data;
}

const showDocument = (docId: string) => {
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");
};

const ConceptGraph: Component<Props> = (props) => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [maxNodes, setMaxNodes] = createSignal(50);

  // now interpreted as percentile threshold
  const [minEdge, setMinEdge] = createSignal(7);

  const [selectedNode, setSelectedNode] =
    createSignal<EventNode | null>(null);

  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);

  const [yearMode, setYearMode] =
    createSignal<"single" | "range">("range");

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = props.data[concept()];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });

  createEffect(() => {
    const [min, max] = yearBounds();

    if (yearMode() === "single") {
      const mid = Math.floor((min + max) / 2);
      setFromYear(mid);
      setToYear(mid);
      return;
    }

    setFromYear(min);
    setToYear(max);
  });

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];

    return filterByYearRange(
      cd.events,
      fromYear(),
      toYear()
    );
  });

  const graphData = createMemo<GraphData>(() => {
    return buildEventGraph(
      yearFiltered(),
      minEdge(),
      maxNodes()
    );
  });

  /* ---------------------------------------------------------------------- */
  /* D3 */
  /* ---------------------------------------------------------------------- */

  let svgRef!: SVGSVGElement;

  let simulationRef:
    | d3.Simulation<EventNode, EventEdge>
    | null = null;

  function render() {
    const { nodes, edges, maxWeight, maxDegree } =
      graphData();

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
        .attr("fill", "rgb(205,89,89)")
        .attr("font-size", "2rem")
        .text("No graph");
      return;
    }

    const nodeRadius =
      d3.scaleSqrt().domain([0, maxDegree]).range([5, 18]);

    const edgeOpacity =
      d3.scaleLinear().domain([0, maxWeight]).range([0.2, 1]);

    const edgeWidth =
      d3.scaleLinear().domain([0, maxWeight]).range([1, 8]);

    const container =
      svg.append("g").attr("class", "zoom-container");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.1, 8])
        .on("zoom", (event) =>
          container.attr("transform", event.transform)
        )
    );

    const edgeSelection = container
      .append("g")
      .selectAll<SVGLineElement, EventEdge>("line")
      .data(edges)
      .join("line")
      .attr("stroke", "#8fb7ff")
      .attr("stroke-opacity", (d) =>
        edgeOpacity(d.weight)
      )
      .attr("stroke-width", (d) =>
        edgeWidth(d.weight)
      );

    const nodeGroup = container
      .append("g")
      .selectAll<SVGGElement, EventNode>("g")
      .data(nodes, (d) => d.id)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => setSelectedNode(d))
      .call(
        d3.drag<SVGGElement, EventNode>()
          .on("start", (event, d) => {
            if (!event.active) {
              simulationRef?.alphaTarget(0.3).restart();
            }

            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on("end", (event, d) => {
            if (!event.active) {
              simulationRef?.alphaTarget(0);
            }

            d.fx = null;
            d.fy = null;
          })
      );

    nodeGroup
      .append("circle")
      .attr("r", (d) => nodeRadius(d.degree))
      .attr("fill", "#d8ebff");

    nodeGroup
      .append("text")
      .text((d) =>
        d.pub_year !== undefined
          ? `${ d.token } (${ d.pub_year })`
          : d.token
      )
      .attr("dy", -14)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "rgba(220,235,255,0.95)");

    simulationRef?.stop();

    simulationRef = d3
      .forceSimulation<EventNode>(nodes)
      .force(
        "link",
        d3.forceLink<EventNode, EventEdge>(edges)
          .id((d) => d.id)
          .distance((d) =>
            Math.max(60, 220 - d.weight * 10)
          )
      )
      .force(
        "charge",
        d3.forceManyBody().strength(-260)
      )
      .force(
        "center",
        d3.forceCenter(W / 2, H / 2)
      )
      .force(
        "collision",
        d3.forceCollide<EventNode>().radius(
          (d) => nodeRadius(d.degree) + 6
        )
      )
      .on("tick", () => {
        edgeSelection
          .attr("x1", (d) => d.source.x ?? 0)
          .attr("y1", (d) => d.source.y ?? 0)
          .attr("x2", (d) => d.target.x ?? 0)
          .attr("y2", (d) => d.target.y ?? 0);

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
  });

  /* ---------------------------------------------------------------------- */
  /* UI */
  /* ---------------------------------------------------------------------- */

  return (
    <div
      style={{
        display: "flex",
        "flex-direction": "column",
        height: "100%",
        width: "100%",
      }}
    >
      <header class="center-align fill max surface-container-low small-padding top-padding">
        <nav>

          <div class="field suffix border middle-align">
            <select
              value={concept()}
              onChange={(e) =>
                setConcept(e.currentTarget.value)
              }
            >
              <For each={concepts}>
                {(c) => <option value={c}>{c}</option>}
              </For>
            </select>
            <output>Concept</output>
          </div>

          <div class="field suffix border middle-align">
            <select
              value={maxNodes()}
              onChange={(e) =>
                setMaxNodes(Number(e.currentTarget.value))
              }
            >
              <For each={[10, 20, 50, 100]}>
                {(n) => <option value={n}>{n}</option>}
              </For>
            </select>
            <output>Max nodes</output>
          </div>

          <div class="field middle-align">
            <div class="slider tiny">
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={minEdge()}
                onInput={(e) =>
                  setMinEdge(
                    Number(e.currentTarget.value)
                  )
                }
              />
              <span />
              <span class="tooltip bottom" />
            </div>

            <output class="small-padding top-padding">
              Overlap percentile {minEdge()}
            </output>
          </div>

        </nav>
      </header>

      <div
        style={{
          display: "flex",
          flex: "1",
          overflow: "hidden",
        }}
        class="background"
      >
        <svg
          ref={svgRef!}
          style={{
            flex: "1",
            display: "block",
          }}
          class="surface-container-lowest"
        />

        <Show when={selectedNode()}>
          <aside
            class="surface-container-high padding border"
            style={{
              width: "22rem",
              "overflow-y": "auto",
            }}
          >
            <Show when={selectedNode()}>
              {(node) => (
                <>
                  <h2>{node().token}</h2>

                  <div>
                    Event ID: {node().id}
                  </div>

                  <div>
                    Year: {node().pub_year ?? "—"}
                  </div>

                  <div>
                    Degree: {node().degree}
                  </div>

                  <Show when={node().doc_id}>
                    <button
                      class="chip small-margin"
                      onClick={() =>
                        showDocument(node().doc_id!)
                      }
                    >
                      {node().doc_id}
                    </button>
                  </Show>

                  <h3 class="top-padding">
                    FAISS neighbours
                  </h3>

                  <For each={node().neighbours}>
                    {(nb) => (
                      <div
                        class="small-margin"
                        style={{
                          display: "flex",
                          "justify-content":
                            "space-between",
                        }}
                      >
                        <span>{nb.token}</span>

                        <span class="small-text">
                          {nb.score.toFixed(4)}
                        </span>
                      </div>
                    )}
                  </For>
                </>
              )}
            </Show>
          </aside>
        </Show>
      </div>

      <footer
        class="fixed max center-align small-padding surface-container-low"
      >
        {graphData().nodes.length} nodes
        {" • "}
        {graphData().edges.length} edges
        {" • "}
        {yearFiltered().length} events
      </footer>
    </div>
  );
};

export default ConceptGraph;
