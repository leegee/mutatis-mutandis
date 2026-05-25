/**
 * ConceptGraph.data.ts
 *
 * Data layer for the ConceptGraph pipeline.
 *
 * aggregateConcept     - O(n²), run once per concept (or filtered pub_year).
 * scanYearRange        - derive min/max pub_year from a ConceptData.
 * filterByYearRange    - filter events to a year window before aggregation.
 * buildGraph           - cheap, D3-facing, reruns on filter change only.
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
export const CORPUS_START_YEAR = 1625;
export const CORPUS_END_YEAR = 1665;

/**
 * Derive the min and max pub_year present in a ConceptData.
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
    const y = event.pub_year;
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
 * pub_year is inline on ConceptEvent (added in tier2_0_concept_events.py),
 * so no external doctoyear mapping is needed.
 *
 * Events with no pub_year are excluded when a filter is active.
 */
export function filterByYearRange(
  events: ConceptEvent[],
  fromYear: number,
  toYear: number
): ConceptEvent[] {
  return events.filter(
    (e) =>
      e.pub_year !== undefined &&
      e.pub_year >= fromYear &&
      e.pub_year <= toYear
  );
}

/**
 * Aggregate raw concept events into a provenance-carrying intermediate.
 *
 * O(n²) in neighbours per event - run once per concept or filtered pub_year
 * and memoised by the caller. Carrying pub_year through onto TokenStats.docs
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

      // Store doc_id to pub_year. If the same doc appears multiple times
      // the year is stable so overwriting is safe.
      if (ni.doc_id) {
        stats.docs.set(
          ni.doc_id,
          ni.pub_year
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
            nj.pub_year
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

  let rawEdgeCount = 0;
  let keptEdgeCount = 0;

  // Edge construction phase
  for (const [tokenA, stats] of agg.byToken) {
    for (const [tokenB, count] of stats.coOccurrences) {
      rawEdgeCount++;

      if (tokenA !== tokenB && count >= minEdgeWeight) {
        filteredEdges.push([tokenA, tokenB, count]);
        keptEdgeCount++;
      }
    }
  }

  console.log("[graph] RAW EDGE PAIRS =", rawEdgeCount);
  console.log("[graph] KEPT EDGES =", keptEdgeCount);
  console.log("[graph] MIN EDGE WEIGHT =", minEdgeWeight);

  // Degree accumulation phase
  const degreeMap = new Map<string, number>();

  for (const [a, b] of filteredEdges) {
    degreeMap.set(a, (degreeMap.get(a) ?? 0) + 1);
    degreeMap.set(b, (degreeMap.get(b) ?? 0) + 1);
  }

  console.log("[graph] DEGREE MAP SIZE =", degreeMap.size);

  // Early exit: no structure survived thresholding
  if (degreeMap.size === 0) {
    console.log("[graph] EMPTY DEGREE MAP → returning EMPTY_GRAPH");
    return EMPTY_GRAPH;
  }

  // Node selection phase
  const sortedNodes = [...degreeMap.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxNodes);

  console.log("[graph] SORTED NODES =", sortedNodes.length);
  console.log("[graph] TOP NODE SAMPLE =", sortedNodes[0]);

  const keepSet = new Set(sortedNodes.map(([id]) => id));
  const nodes: GraphNode[] = sortedNodes.map(([id, degree]) => ({
    id,
    degree,
  }));

  const nodeIndex = new Map(nodes.map((n) => [n.id, n]));

  // Final edge filtering phase
  const edges: GraphEdge[] = filteredEdges
    .filter(([a, b]) => keepSet.has(a) && keepSet.has(b))
    .map(([a, b, weight]) => ({
      source: nodeIndex.get(a)!,
      target: nodeIndex.get(b)!,
      weight,
    }));

  console.log("[graph] FINAL NODES =", nodes.length);
  console.log("[graph] FINAL EDGES =", edges.length);

  // Stability metrics
  const maxWeight = Math.max(1, ...edges.map((e) => e.weight));
  const maxDegree = Math.max(1, ...nodes.map((n) => n.degree));

  console.log("[graph] MAX WEIGHT =", maxWeight);
  console.log("[graph] MAX DEGREE =", maxDegree);

  return { nodes, edges, maxWeight, maxDegree };
}