/**
 * ContextGraph.tsx
 *
 * Tier2 contextual neighbourhood explorer.
 *
 * Aggregated mode:
 *   lexical hubs linked by contextual-neighbour cosine overlap
 *
 * Event mode:
 *   raw event nodes linked to neighbour tokens
 *
 * Intended for semantic drift exploration over Tier2 event projections.
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

const MAX_TOP_N = 20;

type ViewMode = "aggregated" | "events";

interface Neighbour {
  token: string;
  score: number;
  event_id?: number;
  doc_id?: string;
  pub_year?: number;
}

interface ConceptEvent {
  event_id?: number;
  token?: string;
  doc_id?: string;
  pub_year?: number;
  neighbours: Neighbour[];
}

interface ConceptData {
  n_events: number;
  events: ConceptEvent[];
}

export interface Tier2Data {
  [concept: string]: ConceptData;
}

interface Props {
  data: Tier2Data;
}

interface TokenBin {
  token: string;
  eventCount: number;

  /**
   * neighbour token -> frequency across events
   */
  neighbourFreq: Map<string, number>;

  /**
   * neighbour token -> summed score
   */
  neighbourScoreSum: Map<string, number>;

  topNeighbours: Array<{
    token: string;
    freq: number;
    meanScore: number;
  }>;

  docs: Map<string, number | undefined>;
  years: Set<number>;
}

interface ContextNode extends d3.SimulationNodeDatum {
  id: string;
  kind: "hub" | "neighbour" | "event";

  eventCount: number;
  hubDegree: number;
  degree: number;

  token?: string;
  doc_id?: string;
  pub_year?: number;
}

interface HubHubEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-hub";
  weight: number;
}

interface HubNbEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-neighbour";
  weight: number;
}

type AnyEdge = HubHubEdge | HubNbEdge;

interface ContextGraphData {
  nodes: ContextNode[];
  hubHubEdges: HubHubEdge[];
  hubNbEdges: HubNbEdge[];
  allEdges: AnyEdge[];

  maxHubHubWeight: number;
  maxEventCount: number;
  maxHubDegree: number;
}

const CORPUS_START_YEAR = 1625;
const CORPUS_END_YEAR = 1665;

const STYLES = `
.cg-layout {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
}

.cg-main {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.cg-svg {
  flex: 1;
  display: block;
  width: 100%;
  height: 100%;
}

.cg-aside {
  width: 22rem;
  flex-shrink: 0;
  overflow-y: auto;
  padding: 1rem;
}

.cg-node {
  cursor: pointer;
}

.cg-link {
  stroke-opacity: 0.25;
}
`;

function scanYearRange(cd: ConceptData): [number, number] {
  const years = cd.events
    .map(e => e.pub_year)
    .filter((y): y is number => y !== undefined);

  if (!years.length) {
    return [CORPUS_START_YEAR, CORPUS_END_YEAR];
  }

  return [Math.min(...years), Math.max(...years)];
}

function filterByYearRange(
  events: ConceptEvent[],
  from: number,
  to: number,
): ConceptEvent[] {
  return events.filter(
    e =>
      e.pub_year !== undefined &&
      e.pub_year >= from &&
      e.pub_year <= to,
  );
}

/**
 * Aggregates contextual observations by lexical token.
 *
 * This intentionally preserves neighbour-frequency distributions
 * rather than reducing immediately to graph edges.
 *
 * The neighbour distribution itself is the semantic signal.
 */
function aggregateByToken(
  events: ConceptEvent[],
): Map<string, TokenBin> {
  const bins = new Map<string, TokenBin>();

  for (const event of events) {
    const token = event.token;
    if (!token) continue;

    let bin = bins.get(token);

    if (!bin) {
      bin = {
        token,
        eventCount: 0,
        neighbourFreq: new Map(),
        neighbourScoreSum: new Map(),
        topNeighbours: [],
        docs: new Map(),
        years: new Set(),
      };

      bins.set(token, bin);
    }

    bin.eventCount++;

    if (event.doc_id) {
      bin.docs.set(event.doc_id, event.pub_year);
    }

    if (event.pub_year !== undefined) {
      bin.years.add(event.pub_year);
    }

    for (const nb of event.neighbours) {
      bin.neighbourFreq.set(
        nb.token,
        (bin.neighbourFreq.get(nb.token) ?? 0) + 1,
      );

      bin.neighbourScoreSum.set(
        nb.token,
        (bin.neighbourScoreSum.get(nb.token) ?? 0) + nb.score,
      );
    }
  }

  for (const bin of bins.values()) {
    bin.topNeighbours = [...bin.neighbourFreq.entries()]
      .map(([token, freq]) => ({
        token,
        freq,
        meanScore:
          (bin.neighbourScoreSum.get(token) ?? 0) / freq,
      }))
      .sort((a, b) => b.freq - a.freq);
  }

  return bins;
}

/**
 * Sparse cosine similarity over neighbour-frequency vectors.
 *
 * This compares contextual neighbourhood distributions between
 * lexical hubs.
 */
function cosineSimilarity(
  a: Map<string, number>,
  b: Map<string, number>,
): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (const v of a.values()) {
    normA += v * v;
  }

  for (const v of b.values()) {
    normB += v * v;
  }

  for (const [token, av] of a.entries()) {
    const bv = b.get(token);
    if (bv !== undefined) {
      dot += av * bv;
    }
  }

  if (!normA || !normB) return 0;

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function buildContextualGraph(
  bins: Map<string, TokenBin>,
  topN: number,
  minSimilarity: number,
  maxHubs: number,
): ContextGraphData {
  const selectedBins = [...bins.values()]
    .sort((a, b) => b.eventCount - a.eventCount)
    .slice(0, maxHubs);

  const nodes: ContextNode[] = [];
  const hubHubEdges: HubHubEdge[] = [];
  const hubNbEdges: HubNbEdge[] = [];

  const nodeMap = new Map<string, ContextNode>();

  for (const bin of selectedBins) {
    const node: ContextNode = {
      id: bin.token,
      kind: "hub",
      eventCount: bin.eventCount,
      hubDegree: 0,
      degree: 0,
    };

    nodes.push(node);
    nodeMap.set(node.id, node);
  }

  for (let i = 0; i < selectedBins.length; i++) {
    for (let j = i + 1; j < selectedBins.length; j++) {
      const a = selectedBins[i];
      const b = selectedBins[j];

      const sim = cosineSimilarity(
        a.neighbourFreq,
        b.neighbourFreq,
      );

      if (sim < minSimilarity) continue;

      const edge: HubHubEdge = {
        kind: "hub-hub",
        source: nodeMap.get(a.token)!,
        target: nodeMap.get(b.token)!,
        weight: sim,
      };

      hubHubEdges.push(edge);

      nodeMap.get(a.token)!.hubDegree++;
      nodeMap.get(b.token)!.hubDegree++;
    }
  }

  const neighbourMap = new Map<string, ContextNode>();

  for (const bin of selectedBins) {
    const hubNode = nodeMap.get(bin.token)!;

    for (const nb of bin.topNeighbours.slice(0, topN)) {
      let nbNode = neighbourMap.get(nb.token);

      if (!nbNode) {
        nbNode = {
          id: nb.token,
          kind: "neighbour",
          eventCount: 0,
          hubDegree: 0,
          degree: 0,
        };

        neighbourMap.set(nb.token, nbNode);
      }

      hubNbEdges.push({
        kind: "hub-neighbour",
        source: hubNode,
        target: nbNode,
        weight: nb.meanScore,
      });

      hubNode.degree++;
      nbNode.degree++;
    }
  }

  const allNodes = [...nodes, ...neighbourMap.values()];
  const allEdges = [...hubHubEdges, ...hubNbEdges];

  return {
    nodes: allNodes,
    hubHubEdges,
    hubNbEdges,
    allEdges,

    maxHubHubWeight:
      d3.max(hubHubEdges, d => d.weight) ?? 1,

    maxEventCount:
      d3.max(nodes, d => d.eventCount) ?? 1,

    maxHubDegree:
      d3.max(nodes, d => d.hubDegree) ?? 1,
  };
}

function buildPureEventGraph(
  events: ConceptEvent[],
  topN: number,
): ContextGraphData {
  if (!events.length) {
    return {
      nodes: [],
      hubHubEdges: [],
      hubNbEdges: [],
      allEdges: [],
      maxHubHubWeight: 1,
      maxEventCount: 1,
      maxHubDegree: 1,
    };
  }

  const nodes: ContextNode[] = [];
  const hubNbEdges: HubNbEdge[] = [];

  const neighbourMap = new Map<string, ContextNode>();

  events.forEach((event, idx) => {
    const nodeId = `event_${ event.event_id ?? idx }`;

    const node: ContextNode = {
      id: nodeId,
      kind: "event",
      eventCount: 1,
      hubDegree: 0,
      degree: 0,

      token: event.token,
      doc_id: event.doc_id,
      pub_year: event.pub_year,
    };

    nodes.push(node);

    const top = [...event.neighbours]
      .sort((a, b) => b.score - a.score)
      .slice(0, topN);

    for (const nb of top) {
      if (!neighbourMap.has(nb.token)) {
        neighbourMap.set(nb.token, {
          id: nb.token,
          kind: "neighbour",
          eventCount: 0,
          hubDegree: 0,
          degree: 0,
        });
      }

      hubNbEdges.push({
        kind: "hub-neighbour",
        source: node,
        target: neighbourMap.get(nb.token)!,
        weight: nb.score,
      });
    }
  });

  return {
    nodes: [...nodes, ...neighbourMap.values()],
    hubHubEdges: [],
    hubNbEdges,
    allEdges: hubNbEdges,

    maxHubHubWeight: 1,
    maxEventCount: 1,
    maxHubDegree: 1,
  };
}

const showDocument = (docId: string) =>
  window.open(
    `/api/doc/${ docId }`,
    "_blank",
    "noopener,noreferrer",
  );

const ContextGraph: Component<Props> = props => {
  const concepts = Object.keys(props.data);

  const [concept, setConcept] = createSignal(
    concepts[0] ?? "",
  );

  const [viewMode, setViewMode] =
    createSignal<ViewMode>("aggregated");

  const [maxHubs, setMaxHubs] = createSignal(50);
  const [topN, setTopN] = createSignal(5);
  const [minSimilarity, setMinSimilarity] =
    createSignal(0.25);

  const [selectedNode, setSelectedNode] =
    createSignal<string | null>(null);

  const [fromYear, setFromYear] =
    createSignal(CORPUS_START_YEAR);

  const [toYear, setToYear] =
    createSignal(CORPUS_END_YEAR);

  const yearFiltered = createMemo(() => {
    const cd = props.data[concept()];
    if (!cd) return [];

    return filterByYearRange(
      cd.events,
      fromYear(),
      toYear(),
    );
  });

  const tokenBins = createMemo(() =>
    aggregateByToken(yearFiltered()),
  );

  const graphData = createMemo<ContextGraphData>(() =>
    viewMode() === "events"
      ? buildPureEventGraph(
        yearFiltered(),
        topN(),
      )
      : buildContextualGraph(
        tokenBins(),
        topN(),
        minSimilarity(),
        maxHubs(),
      ),
  );

  let svgRef!: SVGSVGElement;

  let simulation:
    | d3.Simulation<ContextNode, AnyEdge>
    | undefined;

  function render() {
    const data = graphData();

    const svg = d3.select(svgRef);

    svg.selectAll("*").remove();

    const width = svgRef.clientWidth || 1200;
    const height = svgRef.clientHeight || 900;

    const g = svg.append("g");

    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.2, 8])
        .on("zoom", e => {
          g.attr("transform", e.transform);
        }),
    );

    simulation?.stop();

    simulation = d3
      .forceSimulation(data.nodes)
      .force(
        "link",
        d3
          .forceLink(data.allEdges)
          .id((d: any) => d.id)
          .distance(d =>
            d.kind === "hub-hub" ? 120 : 60,
          ),
      )
      .force(
        "charge",
        d3.forceManyBody().strength(-180),
      )
      .force(
        "center",
        d3.forceCenter(width / 2, height / 2),
      );

    const link = g
      .append("g")
      .selectAll("line")
      .data(data.allEdges)
      .join("line")
      .attr("class", "cg-link")
      .attr("stroke-width", d =>
        Math.max(1, d.weight * 4),
      )
      .attr("stroke", d =>
        d.kind === "hub-hub"
          ? "#ff8800"
          : "#999999",
      );

    const node = g
      .append("g")
      .selectAll("circle")
      .data(data.nodes)
      .join("circle")
      .attr("class", "cg-node")
      .attr("r", d => {
        if (d.kind === "hub") {
          return 5 + Math.sqrt(d.eventCount);
        }

        if (d.kind === "event") {
          return 4;
        }

        return 3;
      })
      .attr("fill", d => {
        if (d.kind === "hub") return "#ff6600";
        if (d.kind === "event") return "#0066ff";
        return "#777";
      })
      .call(
        d3.drag<any, ContextNode>()
          .on("start", e => {
            if (!e.active) simulation?.alphaTarget(0.3).restart();
          })
          .on("drag", (e, d) => {
            d.fx = e.x;
            d.fy = e.y;
          })
          .on("end", e => {
            if (!e.active) simulation?.alphaTarget(0);
          }),
      )
      .on("click", (_, d) => {
        setSelectedNode(d.id);
      });

    const labels = g
      .append("g")
      .selectAll("text")
      .data(
        data.nodes.filter(
          n =>
            n.kind === "hub" ||
            (n.kind === "neighbour" &&
              n.degree >= 2),
        ),
      )
      .join("text")
      .text(d => d.id)
      .attr("font-size", 11);

    simulation.on("tick", () => {
      link
        .attr("x1", d => (d.source as any).x)
        .attr("y1", d => (d.source as any).y)
        .attr("x2", d => (d.target as any).x)
        .attr("y2", d => (d.target as any).y);

      node
        .attr("cx", d => d.x ?? 0)
        .attr("cy", d => d.y ?? 0);

      labels
        .attr("x", d => (d.x ?? 0) + 8)
        .attr("y", d => (d.y ?? 0) + 4);
    });
  }

  createEffect(render);

  onCleanup(() => {
    simulation?.stop();
  });

  return (
    <>
      <style>{STYLES}</style>

      <div class="cg-layout">
        <header class="small-padding">
          <nav style={{ display: "flex", gap: "0.5rem" }}>
            <select
              value={concept()}
              onChange={e =>
                setConcept(e.currentTarget.value)
              }
            >
              <For each={concepts}>
                {c => (
                  <option value={c}>{c}</option>
                )}
              </For>
            </select>

            <select
              value={viewMode()}
              onChange={e =>
                setViewMode(
                  e.currentTarget.value as ViewMode,
                )
              }
            >
              <option value="aggregated">
                Aggregated
              </option>

              <option value="events">
                Events
              </option>
            </select>

            <input
              type="range"
              min="1"
              max={MAX_TOP_N}
              value={topN()}
              onInput={e =>
                setTopN(+e.currentTarget.value)
              }
            />

            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={minSimilarity()}
              onInput={e =>
                setMinSimilarity(
                  +e.currentTarget.value,
                )
              }
            />
          </nav>
        </header>

        <div class="cg-main">
          <svg
            ref={svgRef!}
            class="cg-svg"
          />

          <Show when={selectedNode()}>
            <aside class="cg-aside">
              <h2>{selectedNode()}</h2>

              <button
                onClick={() =>
                  setSelectedNode(null)
                }
              >
                close
              </button>
            </aside>
          </Show>
        </div>
      </div>
    </>
  );
};

export default ContextGraph;