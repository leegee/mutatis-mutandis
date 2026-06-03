/**
 * ConceptGraph.tsx — @cosmos.gl/graph v2 compatible
 *
 * Key v2 API facts used here
 * --------------------------
 *  • new Graph(div, config)         — takes a div, not a canvas
 *  • setPointPositions(Float32Array)— [x1,y1, x2,y2,…]; omit to let simulation place nodes
 *  • setLinks(Float32Array)         — [srcIdx, tgtIdx, …] (integer indices into the point array)
 *  • setPointColors(Float32Array)   — [r,g,b,a, …] normalised 0-1 per point
 *  • setPointSizes(Float32Array)    — one float per point
 *  • setLinkColors(Float32Array)    — [r,g,b,a, …] per link
 *  • setLinkWidths(Float32Array)    — one float per link
 *  • graph.render()                 — must be called after every data / attribute update
 *  • setConfigPartial(…)            — partial update without resetting everything
 *  • onPointMouseOver(index, pos)   — index into the point array, not a node object
 *  • unpause()                      — was restart() in v1
 *  • Flat config (no nested simulation:{} block)
 *
 * Node types
 * ----------
 *  kind 0  event     — amber
 *  kind 1  neighbour — slate-blue
 *  kind 2  concept   — emerald  (optional)
 *
 * Edge types
 * ----------
 *  kind 0  semantic  — event → neighbour, weight = FAISS cosine score
 *  kind 1  cowindow  — event ↔ event sharing (doc_id, window_id)
 *  kind 2  concept   — event → concept node (only when showConceptNodes=true)
 */

import {
  type Component,
  createEffect,
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

// ── Internal graph data model ─────────────────────────────────────────────────

type NodeKind = 0 | 1 | 2;
export const NODE_KIND = {
  EVENT: 0,
  NEIGHBOUR: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;

interface NodeMeta {
  id: string;       // stable string id used for index lookups
  kind: NodeKind;
  label: string;
  docId?: string;
  pubYear?: number | null;
  windowId?: number | null;
  degree?: number | null;
}

type EdgeKind = 0 | 1 | 2;
export const EDGE_KIND = {
  SEMANTIC: 0,
  COWINDOW: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;

interface EdgeMeta {
  srcIdx: number; // index into nodes array
  kind: EdgeKind;
  tgtIdx: number;
  weight: number; // 0–1
}

interface GraphData {
  nodes: NodeMeta[];
  edges: EdgeMeta[];
}

// ── Colours (normalised 0-1 RGBA) ────────────────────────────────────────────

// kind → [r, g, b, a]
const NODE_RGBA: Record<number, readonly [number, number, number, number]> = {
  [NODE_KIND.EVENT]: [0.99, 0.72, 0.07, 1.0],  // amber   — event
  [NODE_KIND.NEIGHBOUR]: [0.42, 0.58, 0.93, 1.0],  // slate   — neighbour
  [NODE_KIND.CONCEPT]: [0.24, 0.85, 0.56, 1.0],  // emerald — concept
};

const EDGE_RGBA: Record<number, readonly [number, number, number, number]> = {
  [EDGE_KIND.SEMANTIC]: [0.99, 0.72, 0.07, 0.75], // semantic
  [EDGE_KIND.COWINDOW]: [0.70, 0.85, 1.00, 0.90], // cowindow
  [EDGE_KIND.CONCEPT]: [0.24, 0.85, 0.56, 0.80], // concept
};

const HIDDEN_RGBA = [0, 0, 0, 0];

const NODE_SIZE: Record<number, number> = { 0: 6, 1: 3.5, 2: 12 };

// ── DB queries ────────────────────────────────────────────────────────────────

async function fetchGraphData(
  concept: string,
  showConceptNodes: boolean,
): Promise<GraphData> {
  // node id string → array index
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

  // 1. Events ─────────────────────────────────────────────────────────────────
  const eventRows = await execRows(
    `SELECT event_id, token, doc_id, pub_year, window_id
     FROM events
     WHERE concept = ?`,
    [concept],
  );

  for (const row of eventRows) {
    const [event_id, token, doc_id, pub_year, window_id] = row as [
      number, string, string, number | null, number | null,
    ];
    // console.log(token)
    addNode({
      id: `e:${ event_id }`,
      kind: 0,
      label: String(token),
      docId: String(doc_id),
      pubYear: pub_year,
      windowId: window_id,
    });
  }

  // 2. Neighbours + semantic edges ────────────────────────────────────────────
  const neighbourRows = await execRows(
    `SELECT n.event_id, n.neighbour_event_id, n.token, n.doc_id,
            n.pub_year, n.window_id, n.score
     FROM neighbours n
     INNER JOIN events e ON e.event_id = n.event_id
     WHERE e.concept = ?`,
    [concept],
  );

  for (const row of neighbourRows) {
    const [event_id, neighbour_event_id, token, doc_id, pub_year, window_id, score] =
      row as [number, number, string, string, number | null, number | null, number];

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
    });

    const srcIdx = idToIdx.get(`e:${ event_id }`);
    if (srcIdx === undefined) continue; // guard: event must exist

    edges.push({ srcIdx, tgtIdx, kind: 0, weight: Math.max(0, Number(score)) });
  }

  // 2.5 Count neighbours --------------
  const degree = new Uint32Array(nodes.length);
  for (const e of edges) {
    if (e.kind !== EDGE_KIND.SEMANTIC) continue;

    degree[e.srcIdx]++;
    degree[e.tgtIdx]++; // optional if undirected view
  }

  for (let i = 0; i < nodes.length; i++) {
    (nodes[i] as any).degree = degree[i];
  }

  // 3. Co-window edges ─────────────────────────────────────────────────────────
  // Group event nodes by (doc_id, window_id) then connect all pairs (capped)
  const buckets = new Map<string, number[]>(); // key → [nodeIdx, …]
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

  // 4. Concept membership (optional) ──────────────────────────────────────────
  if (showConceptNodes) {
    const cIdx = addNode({ id: `c:${ concept }`, kind: 2, label: concept });
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].kind === NODE_KIND.EVENT) {
        edges.push({ srcIdx: i, tgtIdx: cIdx, kind: 2, weight: 0.5 });
      }
    }
  }

  return { nodes, edges };
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


// ── Float32Array builders ─────────────────────────────────────────────────────

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
      e.kind === EDGE_KIND.COWINDOW ? 1.3 : // cowindow — thin fixed
        e.kind === EDGE_KIND.CONCEPT ? 1.0 : // concept  — thin fixed
          Math.max(0.4, e.weight * 2.0), // semantic — weight-scaled
    ),
  );
}

// ── Overlay components ────────────────────────────────────────────────────────

const Legend: Component<{ showConcept: boolean }> = (props) => (
  <div style={{
    position: "absolute", bottom: "16px", left: "16px",
    background: "rgba(10,12,18,0.88)",
    border: "1px solid rgba(255,255,255,0.07)",
    "border-radius": "8px", padding: "10px 14px",
    "font-family": "'IBM Plex Mono', monospace",
    "font-size": "11px", color: "#c8ccd8",
    "pointer-events": "none", "backdrop-filter": "blur(6px)",
  }}>
    <div style={{ "font-weight": "600", "margin-bottom": "7px", "letter-spacing": "0.06em", color: "#fff" }}>
      LEGEND
    </div>
    <Dot color="#fdb811" label="event" />
    <Dot color="#6b94ee" label="neighbour" />
    {props.showConcept && <Dot color="#3dd98f" label="concept" />}
    <div style={{ "margin-top": "8px", "margin-bottom": "5px", opacity: "0.55", "font-size": "10px", "letter-spacing": "0.08em" }}>
      EDGES
    </div>
    <Line color="rgba(253,184,17,0.5)" label="semantic (FAISS)" />
    <Line color="rgba(180,215,255,0.6)" label="co-window" />
    {props.showConcept && <Line color="rgba(61,217,143,0.4)" label="concept" />}
  </div>
);

const Dot: Component<{ color: string; label: string }> = (p) => (
  <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "3px" }}>
    <div style={{ width: "9px", height: "9px", "border-radius": "50%", background: p.color, "flex-shrink": "0" }} />
    <span>{p.label}</span>
  </div>
);

const Line: Component<{ color: string; label: string }> = (p) => (
  <div style={{ display: "flex", "align-items": "center", gap: "8px", "margin-bottom": "3px" }}>
    <div style={{ width: "22px", height: "2px", "border-radius": "1px", background: p.color, "flex-shrink": "0" }} />
    <span>{p.label}</span>
  </div>
);

const StatsBar: Component<{ data: GraphData }> = (p) => (
  <div style={{
    position: "absolute", top: "16px", right: "16px",
    background: "rgba(10,12,18,0.88)",
    border: "1px solid rgba(255,255,255,0.07)",
    "border-radius": "8px", padding: "10px 14px",
    "font-family": "'IBM Plex Mono', monospace",
    "font-size": "11px", color: "#c8ccd8",
    "pointer-events": "none", "backdrop-filter": "blur(6px)",
  }}>
    <Stat label="events" value={p.data.nodes.filter(n => n.kind === NODE_KIND.EVENT).length} />
    <Stat label="neighbours" value={p.data.nodes.filter(n => n.kind === NODE_KIND.NEIGHBOUR).length} />
    <Stat label="semantic" value={p.data.edges.filter(e => e.kind === EDGE_KIND.SEMANTIC).length} />
    <Stat label="co-window" value={p.data.edges.filter(e => e.kind === EDGE_KIND.COWINDOW).length} />
  </div>
);

const Stat: Component<{ label: string; value: number }> = (p) => (
  <div style={{ display: "flex", "justify-content": "space-between", gap: "20px", "margin-bottom": "2px" }}>
    <span style={{ opacity: "0.5" }}>{p.label}</span>
    <span style={{ color: "#fff", "font-weight": "600" }}>{p.value.toLocaleString()}</span>
  </div>
);

interface TipData { node: NodeMeta; x: number; y: number }

const Tooltip: Component<{ tip: TipData }> = (p) => (
  <div style={{
    position: "absolute",
    left: `${ p.tip.x + 14 }px`, top: `${ p.tip.y - 10 }px`,
    background: "rgba(10,12,18,0.96)",
    border: "1px solid rgba(255,255,255,0.12)",
    "border-radius": "6px", padding: "8px 12px",
    "font-family": "'IBM Plex Mono', monospace",
    "font-size": "11px", color: "#e0e2ec",
    "pointer-events": "none", "max-width": "240px",
    "z-index": "10", "box-shadow": "0 4px 24px rgba(0,0,0,0.55)",
  }}>
    <div style={{ "font-weight": "700", "font-size": "13px", color: "#fff", "margin-bottom": "5px" }}>
      {p.tip.node.label}
    </div>
    {p.tip.node.docId && <div><span style={{ opacity: "0.45" }}>doc  </span>{p.tip.node.docId.slice(0, 20)}</div>}
    {p.tip.node.pubYear && <div><span style={{ opacity: "0.45" }}>year </span>{p.tip.node.pubYear}</div>}
    {p.tip.node.windowId != null && <div><span style={{ opacity: "0.45" }}>win  </span>{p.tip.node.windowId}</div>}
    <div style={{ "margin-top": "4px", opacity: "0.35", "font-size": "10px" }}>{p.tip.node.id}</div>
  </div>
);

// ── Main component ────────────────────────────────────────────────────────────

export interface ConceptGraphProps {
  concept: string;
  showConceptNodes?: boolean;
}

export const ConceptGraph: Component<ConceptGraphProps> = (props) => {
  const showConcept = () => props.showConceptNodes ?? false;
  let initialized = false;

  let divRef!: HTMLDivElement;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let graph: Graph | undefined;

  // Keep a ref to current node list so onPointMouseOver can resolve index → meta
  let nodeMeta: NodeMeta[] = [];

  const [tooltip, setTooltip] = createSignal<TipData | null>(null);

  const [data] = createResource(
    () => [controls.concept, showConcept()] as [string, boolean],
    ([concept, showConcept]) => {
      initialized = false;
      return fetchGraphData(concept, showConcept);
    },
  );

  let mouseClientX = 0;
  let mouseClientY = 0;

  onMount(() => {
    graph = new Graph(divRef, {
      // v2 flat config ────────────────────────────────────────────────────────
      renderLinks: true,
      spaceSize: 1024,
      fitViewOnInit: true,           // Automatically fit when graph is first rendered
      fitViewDelay: 20800,             // Give simulation time to settle before fitting
      fitViewPadding: 0.01,          // 12% padding around the graph (adjust as needed)
      simulationRepulsion: 1.1,
      simulationLinkSpring: 0.45,
      simulationLinkDistance: 65,
      simulationFriction: 0.88,
      simulationGravity: 0.12,
      enableDrag: true,
      randomSeed: 12,
      simulationDecay: 10_000,
      // Colours etc
      backgroundColor: "#0c0e14",
      // pointGreyoutOpacity: 0.5,
      // linkGreyoutOpacity: 0.5,
      // Events
      onPointMouseOver: (index: number) => {
        const node = nodeMeta[index];
        if (!node) return;
        const rect = divRef.getBoundingClientRect();
        // Assuming Cosmos renders at full size of the div
        setTooltip({
          node,
          x: rect.left + mouseClientX,
          y: rect.top + mouseClientY,
        });
      },
      onPointMouseOut: () => setTooltip(null),
      onSimulationEnd: () => graph?.fitView(),
      onPointClick: (index, pos) => console.log('[onPointClick]', index, pos)
    });

    divRef.addEventListener("mousemove", (e: MouseEvent) => {
      mouseClientX = e.clientX;
      mouseClientY = e.clientY;
    });

  });


  // Push data into cosmos whenever the resource resolves
  createEffect(() => {
    const d = data();
    if (!d || !graph) return;

    const yearMode = controls.yearMode;
    const fromYear = controls.fromYear;
    const toYear = controls.toYear;
    const topN = controls.topN;

    const nodeVisible = new Uint8Array(d.nodes.length);
    for (let i = 0; i < d.nodes.length; i++) {
      const n = d.nodes[i];
      const timeOk = isNodeVisibleByTime(n, yearMode, fromYear, toYear);
      const degreeOk = n.kind !== NODE_KIND.EVENT || (n.degree ?? 0) <= topN; // only apply threshold to event nodes
      nodeVisible[i] = timeOk && degreeOk ? 1 : 0;
    }

    graph.setPointColors(buildPointColors(d.nodes, yearMode, fromYear, toYear));
    graph.setPointSizes(buildPointSizes(d.nodes));

    graph.setLinkColors(buildLinkColors(d.edges, nodeVisible));
    graph.setLinkWidths(buildLinkWidths(d.edges));

    if (!initialized) {
      // setPointPositions with no arguments (or omitting) lets the simulation
      // place nodes. We still need to call it to set the node count.
      // Pass a zeroed array of the right size; simulation will move nodes.
      graph.setPointPositions(new Float32Array(d.nodes.length * 2));
      graph.setLinks(buildLinks(d.edges));
      initialized = true;
    }

    graph.render();
  });

  onCleanup(() => {
    graph?.pause?.();
  });

  return (
    <article class="max surface-container" style={{
      position: "relative",
      overflow: "hidden",
      "flex-direction": "column",
      display: "flex",
      flex: 1,
      height: '100%'
    }}>
      <ControlsHeader />
      <div id="cosmos-mount-point" ref={divRef} class="max" style={{
        position: "relative",
        overflow: "hidden",
        height: '100%'
      }}></div>

      <Show when={data.loading}>
        <MsgSettingLayout />
      </Show>

      <Show when={data.error}>
        <div class="error" style={{
          position: "absolute", inset: "0",
          display: "flex", "align-items": "center", "justify-content": "center",
        }}>
          {String(data.error)}
        </div>
      </Show>

      <Show when={!data.loading && !data.error && data()}>
        <StatsBar data={data()!} />
        <Legend showConcept={showConcept()} />
      </Show>

      <Show when={tooltip()}>
        <Tooltip tip={tooltip()!} />
      </Show>

    </article>
  );
};

export default ConceptGraph;

