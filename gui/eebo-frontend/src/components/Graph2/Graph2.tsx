/**
 * ConceptGraph.tsx
 *
 */

import {
  type Component,
  createEffect,
  createMemo,
  createResource,
  createSignal,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { Graph } from "@cosmos.gl/graph";
import { execRows } from "../../services/db";
import ControlsHeader from "../ControlsHeader";
import { controls } from "../../state/controls.store";
import MsgSettingLayout from "../MsgSettingLayout";

import "./style.css";
import Sidebar from "./SideBar";
import type { YearBucket } from "../../types";

// Simple internal graph data model

type NodeKind = 0 | 1 | 2;
export const NODE_KIND = {
  EVENT: 0,
  NEIGHBOUR: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;

export interface NodeMeta {
  id: string;       // stable string id used for index lookups
  kind: NodeKind;
  label: string;
  docId?: string;
  pubYear?: number | null;
  windowId?: number | null;
  degree?: number | null;
  tokenIdx: number;
}

type EdgeKind = 0 | 1 | 2;
export const EDGE_KIND = {
  SEMANTIC: 0,
  COWINDOW: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;

export interface EdgeMeta {
  srcIdx: number; // index into nodes array
  kind: EdgeKind;
  tgtIdx: number;
  weight: number; // 0–1
}

export interface GraphData {
  nodes: NodeMeta[];
  edges: EdgeMeta[];
  years: YearBucket[];
}

// Colours (normalised 0-1 RGBA)

// kind to [r, g, b, a]
const NODE_RGBA: Record<number, readonly [number, number, number, number]> = {
  [NODE_KIND.EVENT]: [1, 0.65, 0.15, 1.0],
  [NODE_KIND.NEIGHBOUR]: [0.35, 0.55, 0.95, 1.0],
  [NODE_KIND.CONCEPT]: [0.2, 0.8, 0.5, 1.0],
};

const EDGE_RGBA: Record<number, readonly [number, number, number, number]> = {
  [EDGE_KIND.SEMANTIC]: [0.99, 0.72, 0.07, 0.5],
  [EDGE_KIND.COWINDOW]: [0.70, 0.85, 1.00, 0.90],
  [EDGE_KIND.CONCEPT]: [0.24, 0.85, 0.56, 0.80],
};

const HIDDEN_RGBA = [0, 0, 0, 0];

const NODE_SIZE: Record<number, number> = {
  [NODE_KIND.EVENT]: 8,
  [NODE_KIND.NEIGHBOUR]: 5,
  [NODE_KIND.CONCEPT]: 12
};

// DB queries
async function loadGraphData(
  concept: string,
  showConceptNodes: boolean,
): Promise<GraphData> {
  console.debug('[graph2 loadGraphData]', concept)
  // node id string to array index
  const idToIdx = new Map<string, number>();
  const nodes: NodeMeta[] = [];
  const edges: EdgeMeta[] = [];

  function addNode(n: NodeMeta): number {
    const existing = idToIdx.get(n.id);
    if (existing !== undefined) return existing;
    const idx = nodes.length;
    nodes.push(n);
    idToIdx.set(n.id, idx);
    return idx;
  }

  // 1. Events
  const eventRows = await execRows(
    `SELECT event_id, token, doc_id, pub_year, window_id, token_idx
     FROM events
     WHERE concept = ?`,
    [concept],
  );

  const yearCounts = new Map<number, number>();

  for (const row of eventRows) {
    const [event_id, token, doc_id, pub_year, window_id, token_idx] = row as [
      number, string, string, number | null, number | null, number
    ];
    // console.debug(token)
    addNode({
      id: `e:${ event_id }`,
      kind: NODE_KIND.EVENT,
      label: String(token),
      docId: String(doc_id),
      pubYear: pub_year,
      windowId: window_id,
      tokenIdx: token_idx
    });
    yearCounts.set(pub_year || 0, (yearCounts.get(pub_year || 0) ?? 0) + 1);
  }

  const years = [...yearCounts.entries()]
    .sort(([a], [b]) => a - b)
    .map(([year, count]) => ({
      year,
      count,
    }));

  // 2. Neighbours + semantic edges
  const neighbourRows = await execRows(
    `SELECT n.event_id, n.neighbour_event_id, n.token, n.doc_id,
            n.pub_year, n.window_id, n.score, n.token_idx
     FROM neighbours n
     INNER JOIN events e ON e.event_id = n.event_id
     WHERE e.concept = ?`,
    [concept],
  );

  for (const row of neighbourRows) {
    const [event_id, neighbour_event_id, token, doc_id, pub_year, window_id, score, token_idx] =
      row as [number, number, string, string, number | null, number | null, number, number];

    // Prefer the event-node id if this neighbour is also a concept event
    const nStringId = idToIdx.has(`e:${ neighbour_event_id }`)
      ? `e:${ neighbour_event_id }`
      : `n:${ neighbour_event_id }`;

    const tgtIdx = addNode({
      id: nStringId,
      kind: idToIdx.has(`e:${ neighbour_event_id }`) ? 0 : 1,
      label: String(token),
      docId: String(doc_id),
      pubYear: pub_year,
      windowId: window_id,
      tokenIdx: token_idx
    });

    const srcIdx = idToIdx.get(`e:${ event_id }`);
    if (srcIdx === undefined) continue; // guard: event must exist

    edges.push({ srcIdx, tgtIdx, kind: 0, weight: Math.max(0, Number(score)) });
  }

  // 3. Count neighbours
  const degree = new Uint32Array(nodes.length);
  for (const e of edges) {
    if (e.kind !== EDGE_KIND.SEMANTIC) continue;

    degree[e.srcIdx]++;
    degree[e.tgtIdx]++; // optional if undirected view
  }

  for (let i = 0; i < nodes.length; i++) {
    (nodes[i] as any).degree = degree[i];
  }

  // 4. Co-window edges
  // Group event nodes by (doc_id, window_id) then connect all pairs (capped)
  const buckets = new Map<string, number[]>(); // key to [nodeIdx, …]
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.kind !== 0 || n.docId == null || n.windowId == null) continue;
    const key = `${ n.docId }::${ n.windowId }`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(i);
  }

  for (const members of buckets.values()) {
    if (members.length < 2) continue;
    const capped = members.slice(0, 6); // cap clique size
    for (let a = 0; a < capped.length; a++) {
      for (let b = a + 1; b < capped.length; b++) {
        edges.push({ srcIdx: capped[a], tgtIdx: capped[b], kind: 1, weight: 1.0 });
      }
    }
  }

  // 5. Concept membership
  if (showConceptNodes) {
    const cIdx = addNode({
      id: `c:${ concept }`,
      kind: NODE_KIND.CONCEPT,
      label: concept,
      tokenIdx: -1 // TODO what?
    });
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].kind === NODE_KIND.EVENT) {
        edges.push({ srcIdx: i, tgtIdx: cIdx, kind: 2, weight: 0.5 });
      }
    }
  }

  return { nodes, edges, years };
}

function isNodeVisibleByTime(
  n: NodeMeta,
  yearMode: string,
  fromYear: number,
  toYear: number
): boolean {
  if (yearMode === 'single') {
    return fromYear === n.pubYear;
  }

  return (
    n.pubYear != null &&
    n.pubYear >= fromYear &&
    n.pubYear <= toYear
  );
}


// Float32Array builders

function buildPointColors(
  nodes: NodeMeta[],
  yearMode: string,
  fromYear: number,
  toYear: number
): Float32Array {
  const buf = new Float32Array(nodes.length * 4);

  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];

    const visible = isNodeVisibleByTime(n, yearMode, fromYear, toYear);

    const rgba = visible
      ? (NODE_RGBA[n.kind] ?? NODE_RGBA[1])
      : HIDDEN_RGBA;

    buf[i * 4] = rgba[0];
    buf[i * 4 + 1] = rgba[1];
    buf[i * 4 + 2] = rgba[2];
    buf[i * 4 + 3] = rgba[3];
  }

  return buf;
}

function buildPointSizes(nodes: NodeMeta[]): Float32Array {
  return new Float32Array(nodes.map((n) => NODE_SIZE[n.kind] ?? 4));
}

function buildLinks(edges: EdgeMeta[]): Float32Array {
  const buf = new Float32Array(edges.length * 2);
  for (let i = 0; i < edges.length; i++) {
    buf[i * 2] = edges[i].srcIdx;
    buf[i * 2 + 1] = edges[i].tgtIdx;
  }
  return buf;
}

function buildLinkColors(
  edges: EdgeMeta[],
  nodeVisible: Uint8Array
): Float32Array {
  const buf = new Float32Array(edges.length * 4);

  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];

    const visible = nodeVisible[e.srcIdx] && nodeVisible[e.tgtIdx];

    const rgba = visible
      ? EDGE_RGBA[e.kind] ?? EDGE_RGBA[0]
      : HIDDEN_RGBA;

    buf[i * 4] = rgba[0];
    buf[i * 4 + 1] = rgba[1];
    buf[i * 4 + 2] = rgba[2];
    buf[i * 4 + 3] = rgba[3];
  }

  return buf;
}

function buildLinkWidths(edges: EdgeMeta[]): Float32Array {
  return new Float32Array(
    edges.map((e) =>
      e.kind === EDGE_KIND.COWINDOW ? 2 :
        e.kind === EDGE_KIND.CONCEPT ? 2 :
          Math.max(1, e.weight * 2.0), // weight-scaled
    ),
  );
}

// Overlay components

const Dot: Component<{ color: string; label: string }> = (p) => (
  <div style={{ display: "inline-flex", "align-items": "center", gap: "8px", "margin-bottom": "3px" }}>
    <div style={{ width: "0.7rem", height: "0.7rem", "border-radius": "50%", background: p.color, "flex-shrink": "0" }} />
    <span>{p.label}</span>
  </div>
);

const Line: Component<{ color: string; label: string }> = (p) => (
  <div style={{ display: "inline-flex", "align-items": "center", gap: "8px", "margin-bottom": "3px" }}>
    <div style={{ width: "1rem", height: "2px", "border-radius": "1px", background: p.color, "flex-shrink": "0" }} />
    <span>{p.label}</span>
  </div>
);

interface TipData { node: NodeMeta; x: number; y: number }

const Tooltip: Component<{ tip: TipData }> = (p) => (
  <aside class="surface-container-highest border padding large-elevate" style={{
    position: "absolute",
    left: `${ p.tip.x + 14 }px`,
    top: `${ p.tip.y - 10 }px`,
    "max-width": "240px",
    "z-index": "10",
  }}>
    <h6> {p.tip.node.label} </h6>
    {p.tip.node.docId && <div><span style={{ opacity: "0.45" }}>doc  </span>{p.tip.node.docId.slice(0, 20)}</div>}
    {p.tip.node.pubYear && <div><span style={{ opacity: "0.45" }}>year </span>{p.tip.node.pubYear}</div>}
    {p.tip.node.windowId != null && <div><span style={{ opacity: "0.45" }}>win  </span>{p.tip.node.windowId}</div>}
    <div style={{ "margin-top": "4px", opacity: "0.35", "font-size": "10px" }}>{p.tip.node.id}</div>
  </aside>
);

// Main component

export interface ConceptGraphProps {
  showConceptNodes?: boolean;
}

export const ConceptGraph: Component<ConceptGraphProps> = (props) => {
  const showConcept = () => props.showConceptNodes ?? true;
  const [selectedNode, setSelectedNode] = createSignal<NodeMeta | null>(null);
  const [graphProgress, setGraphProgress] = createSignal(0);

  let divRef!: HTMLDivElement;
  let graph: Graph | undefined;

  // Keep a ref to current node list so onPointMouseOver can resolve index to meta
  let nodeMeta: NodeMeta[] = [];

  const [tooltip, setTooltip] = createSignal<TipData | null>(null);

  const [data] = createResource(
    () => [controls.concept, showConcept()] as [string, boolean],
    ([concept, showConcept]) => {
      if (graph) graph.destroy();
      return loadGraphData(concept, showConcept);
    },
  );

  let mouseClientX = 0;
  let mouseClientY = 0;

  function setGraph() {
    if (graph) {
      console.debug("[graph2.setGraph] destroy old graph");
      graph.destroy();
    }

    graph = new Graph(divRef, {
      renderLinks: true,
      spaceSize: 2048 * 2,
      fitViewOnInit: true,           // Automatically fit when graph is first rendered
      fitViewDelay: 1000,            // Give simulation time to settle before fitting
      fitViewPadding: 0.01,          // 12% padding around the graph (adjust as needed)
      simulationRepulsion: 0.8,
      simulationLinkSpring: 0.45,
      simulationLinkDistance: 65,
      simulationFriction: 0.9,     // slow down
      simulationGravity: 0.12,
      enableDrag: true,
      // randomSeed: 12,
      simulationDecay: 500,
      // Colours etc
      backgroundColor: "#0c0e14",
      pointGreyoutOpacity: 0.1,
      linkGreyoutOpacity: 0.1,

      // Events
      onPointMouseOver: (index: number) => {
        const node = nodeMeta[index];
        if (!node) return;
        setTooltip({ node, x: mouseClientX, y: mouseClientY });
      },
      onPointMouseOut: () => setTooltip(null),
      onSimulationEnd: () => graph?.fitView(),
      onSimulationTick: () => setGraphProgress(graph?.progress || 0),

      onClick: (index, pos) => {
        console.debug('[onClick]', index, pos);
        if (index) {
          setSelectedNode(nodeMeta[index]);
          graph?.selectPointByIndex(index);
        } else {
          setSelectedNode(null);
          graph?.unselectPoints();

        }
      }
    });

    console.debug("[graph2.setGraph] created new graph");
  }

  onMount(() => {
    console.debug("[graph2.onMount] enter");
    divRef.addEventListener("mousemove", (e: MouseEvent) => {
      const rect = divRef.getBoundingClientRect();
      mouseClientX = e.clientX - rect.left;
      mouseClientY = e.clientY - rect.top;
    });
  });

  // Displayed in footer
  const counts = createMemo(() => {
    const d = data();
    if (!d) return null;
    return {
      yearBuckets: data()?.years,
      events: d.nodes.filter(n => n.kind === NODE_KIND.EVENT).length,
      neighbours: d.nodes.filter(n => n.kind === NODE_KIND.NEIGHBOUR).length,
      concepts: d.nodes.filter(n => n.kind === NODE_KIND.CONCEPT).length,
      semantic: d.edges.filter(e => e.kind === EDGE_KIND.SEMANTIC).length,
      cowindow: d.edges.filter(e => e.kind === EDGE_KIND.COWINDOW).length,
      conceptEdges: d.edges.filter(e => e.kind === EDGE_KIND.CONCEPT).length,
    };
  });

  // Effect 1: structural - runs only when data (concept) changes
  createEffect(() => {
    const d = data();
    if (!d) return;
    if (data.loading || data.error) return;
    if (d !== data.latest) return;

    if (graph) {
      console.debug("[graph2.effect 1] reset graph");
      setGraphProgress(0);
      setTooltip(null);
      setSelectedNode(null);
      graph?.unselectPoints();
    }

    setGraph()

    console.debug("[graph2.effect 1] set graph structure");
    graph!.setPointPositions(new Float32Array(d.nodes.length * 2));
    graph!.setLinks(buildLinks(d.edges));
    graph!.setPointSizes(buildPointSizes(d.nodes));
    graph!.setLinkWidths(buildLinkWidths(d.edges));
    nodeMeta = d.nodes;

    graph!.render();
    graph!.unpause();
    console.debug("[graph2.effect 1] unpause");
  });

  // Effect 2: visual-only — runs when filters change, no simulation reset --
  createEffect(() => {
    const d = data();
    if (!d || !graph) return;
    if (data.loading || data.error) return;
    if (d !== data.latest) return;

    console.debug("[graph2.effect 2] enter");

    const { yearMode, fromYear, toYear, topN } = controls;

    const nodeVisible = new Uint8Array(d.nodes.length);
    for (let i = 0; i < d.nodes.length; i++) {
      const n = d.nodes[i];
      const timeOk = isNodeVisibleByTime(n, yearMode, fromYear, toYear);
      const degreeOk = n.kind !== NODE_KIND.EVENT || (n.degree ?? 0) <= topN;
      nodeVisible[i] = timeOk && degreeOk ? 1 : 0;
    }

    graph.setPointColors(buildPointColors(d.nodes, yearMode, fromYear, toYear));
    graph.setLinkColors(buildLinkColors(d.edges, nodeVisible));
    graph.render();
    console.debug("[graph2.effect 2] rendered");
  });

  onCleanup(() => {
    console.debug("[graph2.cleanup] enter");
    graph?.pause?.();
  });

  return (
    <article class="max surface-container" style={{
      position: "relative",
      overflow: "hidden",
      display: "flex",
      "flex-direction": "column",
      flex: 1,
      height: '100%'
    }}>

      <Show when={!data.loading}>
        <ControlsHeader />
        <Show when={graphProgress() < 1}>
          <progress max={1} value={graphProgress()} />
        </Show>
      </Show>

      <div id="graph_sidebar_row" style={{
        display: "flex",
        flex: 1,
        overflow: "hidden",
      }}>
        <div id="cosmos-mount-point" ref={divRef} class="max" style={{
          position: "relative",
          overflow: "hidden",
          flex: 1,
          height: '100%'
        }}></div>


        <Show when={selectedNode()}>
          <Sidebar
            selectedNode={selectedNode()}
            graphData={data() ?? null}
            onClose={() => setSelectedNode(null)}
          />
        </Show>
      </div>

      <Show when={data.loading}>
        <MsgSettingLayout />
      </Show>

      <Show when={data.error}>
        <div class="error-container border">
          {String(data.error)}
        </div>
      </Show>

      <Show when={!data.loading && !data.error && data() && counts()}>
        <footer class="surface-container-low">
          <nav class="padding">
            <div style="display:flex;gap:1em">
              <Dot color={`rgba(${ NODE_RGBA[NODE_KIND.EVENT].map(_ => _ * 255).join(",") })`}
                label={`events (${ counts()!.events.toLocaleString() })`}
              />
              <Dot color={`rgba(${ NODE_RGBA[NODE_KIND.NEIGHBOUR].map(_ => _ * 255).join(",") })`}
                label={`neighbours (${ counts()!.neighbours.toLocaleString() })`}
              />
              <Dot color={`rgba(${ NODE_RGBA[NODE_KIND.CONCEPT].map(_ => _ * 255).join(",") })`}
                label={`concepts (${ counts()!.concepts.toLocaleString() })`}
              />

              {/* <hr class="vertical" /> */}
              {" "}&middot;{" "}

              <Line color={`rgba(${ EDGE_RGBA[EDGE_KIND.SEMANTIC].map(_ => _ * 255).join(",") })`}
                label={`semantic (${ counts()!.semantic.toLocaleString() })`}
              />
              <Line color={`rgba(${ EDGE_RGBA[EDGE_KIND.COWINDOW].map(_ => _ * 255).join(",") })`}
                label={`co-window (${ counts()!.cowindow.toLocaleString() })`}
              />
              <Line color={`rgba(${ EDGE_RGBA[EDGE_KIND.CONCEPT].map(_ => _ * 255).join(",") })`}
                label={`concept (${ counts()!.conceptEdges.toLocaleString() })`}
              />
            </div>

            <div class="max"></div>
            <div class="medium-opacity">
              <h1>
                <Show when={controls.yearMode === 'single'}>
                  {controls.fromYear}
                </Show>
                <Show when={controls.yearMode === 'range'}>
                  {controls.fromYear} &mdash; {controls.toYear}
                </Show>
              </h1>
            </div>
          </nav>

        </footer>
      </Show>

      <Show when={tooltip()}>
        <Tooltip tip={tooltip()!} />
      </Show>

    </article>
  );
};

export default ConceptGraph;

