/**
 * ConceptGraph.data.ts
 *
 * Data layer for the ConceptGraph pipeline.
 *
 * aggregateConcept     — O(n²), run once per concept (or filtered slice).
 * scanYearRange        — derive min/max pub_year (or slice_start) from a ConceptData.
 * filterByYearRange    — filter events to a year window before aggregation.
 * buildGraph           — cheap, D3-facing, reruns on filter change only.
 */

import type {
  ConceptData,
  ConceptEvent,
  AggregatedConcept,
  TokenStats,
  GraphData,
  GraphNode,
  GraphEdge,
} from "./ConceptGraph.types";


// Year range scanning

export const SLICES = [
  [1625, 1629],
  [1630, 1634],
  [1635, 1639],
  [1640, 1640],
  [1641, 1641],
  [1642, 1642],
  [1643, 1643],
  [1644, 1644],
  [1645, 1645],
  [1646, 1646],
  [1647, 1647],
  [1648, 1648],
  [1649, 1649],
  [1650, 1650],
  [1651, 1651],
  [1652, 1654],
  [1655, 1657],
  [1658, 1660],
  [1661, 1665],
];

export const SLICE_MIN = SLICES[0][0];
export const SLICE_MAX = SLICES[SLICES.length - 1][1];

/**
 * Derive the min and max pub_year/slice_start present in a ConceptData.
 *
 * Called once at load time so the UI can set slider bounds.
 * Returns [undefined, undefined] if no year data is present.
 */
export function scanYearRange(
  conceptData: ConceptData
): [number | undefined, number | undefined] {
  let min: number | undefined;
  let max: number | undefined;

  for (const event of conceptData.events) {
    const y = event.slice_start; // event.pub_year;
    if (y === undefined) continue;
    if (min === undefined || y < min) min = y;
    if (max === undefined || y > max) max = y;
  }

  return [min, max];
}


// Temporal filtering


/**
 * Filter concept events to a publication year range.
 *
 * pub_year/slice_start is inline on ConceptEvent (added in tier2_0_concept_events.py),
 * so no external doctoyear mapping is needed.
 *
 * Events with no pub_year/slice_start are excluded when a filter is active.
 */
export function filterByYearRange(
  events: ConceptEvent[],
  fromYear: number,
  toYear: number
): ConceptEvent[] {
  return events.filter(
    (e) =>
      e.slice_start !== undefined &&
      e.slice_start >= fromYear &&
      e.slice_start <= toYear
    // e.pub_year !== undefined &&
    // e.pub_year >= fromYear &&
    // e.pub_year <= toYear
  );
}


// Aggregation


/**
 * Aggregate raw concept events into a provenance-carrying intermediate.
 *
 * O(n²) in neighbours per event — run once per concept or filtered slice
 * and memoised by the caller. Carrying pub_year/slice_start through onto TokenStats.docs
 * means the drill-down panel can show document years without re-scanning.
 */
export function aggregateConcept(
  conceptData: ConceptData,
  events?: ConceptEvent[]
): AggregatedConcept {
  const src = events ?? conceptData.events;
  const byToken = new Map<string, TokenStats>();

  function getOrCreate(token: string): TokenStats {
    if (!byToken.has(token)) {
      byToken.set(token, {
        token,
        coOccurrences: new Map(),
        docs: new Map(),
        totalAppearances: 0,
      });
    }
    return byToken.get(token)!;
  }

  for (const event of src) {
    const neighbours = event.neighbours;

    for (let i = 0; i < neighbours.length; i++) {
      const ni = neighbours[i];
      const stats = getOrCreate(ni.token);

      stats.totalAppearances += 1;

      // Store doc_id to pub_year/slice_start. If the same doc appears multiple times
      // the year is stable so overwriting is safe.
      if (ni.doc_id) {
        stats.docs.set(
          ni.doc_id,
          // ni.pub_year
          ni.slice_start
        );
      }

      for (let j = i + 1; j < neighbours.length; j++) {
        const nj = neighbours[j];

        const countIJ = stats.coOccurrences.get(nj.token) ?? 0;
        stats.coOccurrences.set(nj.token, countIJ + 1);

        const njs = getOrCreate(nj.token);
        const countJI = njs.coOccurrences.get(ni.token) ?? 0;
        njs.coOccurrences.set(ni.token, countJI + 1);

        if (nj.doc_id) {
          njs.docs.set(
            nj.doc_id,
            // nj.pub_year
            nj.slice_start
          );
        }
      }
    }
  }

  return { byToken, nEvents: src.length };
}


// Graph construction (D3-facing)


export const EMPTY_GRAPH: GraphData = {
  nodes: [],
  edges: [],
  maxWeight: 1,
  maxDegree: 1,
};

/**
 * Build D3-ready graph data from an AggregatedConcept.
 *
 * Cheap: no event scanning. GraphNode carries no provenance;
 * for drill-down, callers query AggregatedConcept.byToken[node.id] directly.
 */
export function buildGraph(
  agg: AggregatedConcept,
  minEdgeWeight: number,
  maxNodes: number
): GraphData {
  const filteredEdges: Array<[string, string, number]> = [];

  for (const [tokenA, stats] of agg.byToken) {
    for (const [tokenB, count] of stats.coOccurrences) {
      if (tokenA < tokenB && count >= minEdgeWeight) {
        filteredEdges.push([tokenA, tokenB, count]);
      }
    }
  }

  const degreeMap = new Map<string, number>();

  for (const [a, b] of filteredEdges) {
    degreeMap.set(a, (degreeMap.get(a) ?? 0) + 1);
    degreeMap.set(b, (degreeMap.get(b) ?? 0) + 1);
  }

  if (degreeMap.size === 0) return EMPTY_GRAPH;

  const sortedNodes = [...degreeMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxNodes);

  const keepSet = new Set(sortedNodes.map(([id]) => id));
  const nodes: GraphNode[] = sortedNodes.map(([id, degree]) => ({ id, degree }));
  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  const edges: GraphEdge[] = filteredEdges
    .filter(([a, b]) => keepSet.has(a) && keepSet.has(b))
    .map(([a, b, weight]) => ({
      source: nodeIndex.get(a)!,
      target: nodeIndex.get(b)!,
      weight,
    }));

  const maxWeight = Math.max(1, ...edges.map((e) => e.weight));
  const maxDegree = Math.max(1, ...nodes.map((n) => n.degree));

  return { nodes, edges, maxWeight, maxDegree };
}
