import { execRows } from "../../services/db";
import { type EdgeMeta, type NodeMeta, type Graph2Data, EDGE_KIND, NODE_KIND } from "./types";

export async function loadGraphData(
  concept: string
): Promise<Graph2Data> {
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
    `SELECT event_id, token, doc_id, pub_year, window_id, token_idx, nx, ny
     FROM events
     WHERE concept = ?`,
    [concept],
  );

  // console.debug("eventRows for", concept, eventRows.length);

  const yearCounts = new Map<number, number>();

  for (const row of eventRows) {
    const [event_id, token, doc_id, pub_year, window_id, token_idx, nx, ny] = row as [
      string, string, string, number | null, number | null, number, number, number
    ];

    addNode({
      id: `e:${ event_id }`,
      kind: NODE_KIND.EVENT,
      label: String(token),
      docId: String(doc_id),
      pubYear: pub_year,
      windowId: window_id,
      tokenIdx: token_idx,
      x: nx,
      y: ny,
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
            n.pub_year, n.window_id, n.score, n.token_idx, n.nx, n.ny
     FROM neighbours n
     INNER JOIN events e ON e.event_id = n.event_id
     WHERE e.concept = ?`,
    [concept],
  );

  for (const row of neighbourRows) {
    const [event_id, neighbour_event_id, token, doc_id, pub_year, window_id, score, token_idx, nx, ny] =
      row as [string, string, string, string, number | null, number | null, number, number, number, number];

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
      tokenIdx: token_idx,
      x: nx,
      y: ny,
    });

    const srcIdx = idToIdx.get(`e:${ event_id }`);
    if (srcIdx === undefined) continue; // event must exist

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
  // Group event nodes by (doc_id, window_id) then connect all pairs (capped?)
  const buckets = new Map<string, number[]>(); // key to [nodeIdx, …]
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    if (n.kind !== 0 || n.docId == null || n.windowId == null) continue;
    const key = `${ n.docId }::${ n.windowId }`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key)!.push(i);
  }

  for (const members of buckets.values()) {
    // console.debug("bucket size", members.length);
    if (members.length < 2) continue;
    const capped = members; // members.slice(0, 6); // cap clique size?
    for (let a = 0; a < capped.length; a++) {
      for (let b = a + 1; b < capped.length; b++) {
        edges.push({ srcIdx: capped[a], tgtIdx: capped[b], kind: 1, weight: 1.0 });
      }
    }
  }

  // 5. Concept membership
  if (true) {
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

  // console.log("[ConceptGraph debug]");
  // console.log("nodes:", nodes.length);
  // console.log("edges:", edges.length);

  // console.log(
  //   "node kinds:",
  //   nodes.reduce((a, n) => {
  //     a[n.kind] = (a[n.kind] ?? 0) + 1;
  //     return a;
  //   }, {} as Record<number, number>)
  // );

  // console.log(
  //   "edge kinds:",
  //   edges.reduce((a, e) => {
  //     a[e.kind] = (a[e.kind] ?? 0) + 1;
  //     return a;
  //   }, {} as Record<number, number>)
  // );

  // console.log(
  //   "coordinates:",
  //   {
  //     x: [
  //       Math.min(...nodes.map(n => n.x ?? Infinity)),
  //       Math.max(...nodes.map(n => n.x ?? -Infinity)),
  //     ],
  //     y: [
  //       Math.min(...nodes.map(n => n.y ?? Infinity)),
  //       Math.max(...nodes.map(n => n.y ?? -Infinity)),
  //     ],
  //   }
  // );

  // console.table(
  //   nodes.slice(0, 20).map(n => ({
  //     id: n.id,
  //     kind: n.kind,
  //     x: n.x,
  //     y: n.y
  //   }))
  // );

  return { nodes, edges, years };
}
