import { execRows } from "../../services/db";
import { type EdgeMeta, type NodeMeta, type FDGData, EDGE_KIND, NODE_KIND } from "./types";

export async function loadFDGData(
  concept: string
): Promise<FDGData> {
  console.debug('[loadFDGData loadGraphData]', concept)
  // node id string to array index
  const idToIdx = new Map<string, number>();
  const nodes: NodeMeta[] = [];
  const edges: EdgeMeta[] = [];

  const tokenCounts = new Map<number, Map<string, number>>();

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
  const neighbourRows = await execRows(`
    SELECT
      n.event_id, n.neighbour_event_id, e.token, e.doc_id,
      e.pub_year, e.window_id, n.score, e.token_idx, e.nx, e.ny
    FROM neighbours n
    JOIN events e ON e.event_id = n.neighbour_event_id
    JOIN events src ON src.event_id = n.event_id
    WHERE src.concept = ?
   `,
    [concept],
  );

  for (const row of neighbourRows) {
    const [
      event_id, neighbour_event_id, token, doc_id, pub_year, window_id, score, token_idx, nx, ny
    ] = row as [
      string, string, string, string, number | null, number | null, number, number, number, number
    ];

    const tgtIdx = addNode({
      id: `e:${ neighbour_event_id }`,
      kind: NODE_KIND.EVENT,
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

    const tokenMap = tokenCounts.get(srcIdx) ?? new Map<string, number>();
    tokenMap.set(token, (tokenMap.get(token) ?? 0) + 1);
    tokenCounts.set(srcIdx, tokenMap);

    edges.push({
      srcIdx,
      tgtIdx,
      kind: EDGE_KIND.SEMANTIC,
      weight: Math.max(0, Number(score))
    });
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
    if (
      n.kind !== NODE_KIND.EVENT ||
      n.docId == null ||
      n.windowId == null
    )
      continue;

    const key = `${ n.docId }::${ n.windowId } `;
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
      id: `c:${ concept } `,
      kind: NODE_KIND.CONCEPT,
      label: concept,
      tokenIdx: -1 // No token Idx but required by the type
    });
    for (let i = 0; i < nodes.length; i++) {
      if (nodes[i].kind === NODE_KIND.EVENT) {
        edges.push({ srcIdx: i, tgtIdx: cIdx, kind: 2, weight: 0.5 });
      }
    }
  }

  for (const [idx, counts] of tokenCounts) {
    nodes[idx].neighbourTokens = [...counts.entries()]
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }

  console.log(
    "[loadFDGData] sample semantic fields",
    [...tokenCounts.entries()]
      .slice(0, 5)
      .map(([idx]) => ({
        idx,
        label: nodes[idx].label,
        tokens: nodes[idx].neighbourTokens,
      }))
  );

  return { nodes, edges, years };
}
