/**
 * ConceptGraph.data.ts
 *
 * Data layer for the ConceptGraph pipeline.
 *
 * Responsibilities
 * ----------------
 * aggregateConcept  - O(n²) pass over raw events, run once per concept.
 *                     Produces AggregatedConcept with full provenance.
 *
 * filterByYearRange - Filters raw events by publication year before
 *                     aggregation, enabling temporal split view.
 *
 * buildGraph        - Takes AggregatedConcept + filter params, produces
 *                     GraphData for D3. Cheap: no event scanning.
 *
 * Separation rationale
 * --------------------
 * aggregateConcept and buildGraph are deliberately separate because they
 * serve different consumers:
 *
 *   - Temporal split view calls aggregateConcept twice with different
 *     year-filtered event sets, then buildGraph twice. The aggregation
 *     cost is paid once per filtered slice, not per filter change.
 *
 *   - Document drill-down queries AggregatedConcept.byToken[node.id].docs
 *     directly, without touching GraphData. No provenance needs to live
 *     on GraphNode.
 *
 *   - Filter changes (min edge, max nodes) only rerun buildGraph - the
 *     cheap pass - not aggregateConcept.
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

/**
 * Aggregate raw concept events into a provenance-carrying intermediate.
 *
 * O(n²) in neighbours per event - the same complexity as the original
 * buildGraph, but run once and memoised rather than on every filter change.
 */
export function aggregateConcept(
  conceptData: ConceptData,
  events?: ConceptEvent[]   // optional pre-filtered event list for temporal split
): AggregatedConcept {
  const src = events ?? conceptData.events;
  const byToken = new Map<string, TokenStats>();

  function getOrCreate(token: string): TokenStats {
    if (!byToken.has(token)) {
      byToken.set(token, {
        token,
        coOccurrences: new Map(),
        docs: new Set(),
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

      if (ni.doc_id) stats.docs.add(ni.doc_id);

      // Record pairwise co-occurrence with all other neighbours in this event
      for (let j = i + 1; j < neighbours.length; j++) {
        const nj = neighbours[j];

        const countIJ = stats.coOccurrences.get(nj.token) ?? 0;
        stats.coOccurrences.set(nj.token, countIJ + 1);

        // Symmetric: also record from nj's perspective
        const njs = getOrCreate(nj.token);
        const countJI = njs.coOccurrences.get(ni.token) ?? 0;
        njs.coOccurrences.set(ni.token, countJI + 1);

        if (nj.doc_id) njs.docs.add(nj.doc_id);
      }
    }
  }

  return { byToken, nEvents: src.length };
}

/**
 * Filter concept events to a publication year range.
 *
 * Requires doc_id → year mapping, which the caller must supply.
 * This keeps the data layer independent of any Postgres/API coupling.
 *
 * Usage:
 *   const docYears = await fetchDocYears();   // Map<doc_id, year>
 *   const eventsA = filterByYearRange(conceptData.events, docYears, 1580, 1620);
 *   const aggA = aggregateConcept(conceptData, eventsA);
 *   const graphA = buildGraph(aggA, minEdge, maxNodes);
 */
export function filterByYearRange(
  events: ConceptEvent[],
  fromYear: number,
  toYear: number
): ConceptEvent[] {
  return events.filter(
    (e) => e.pub_year !== undefined &&
      e.pub_year >= fromYear &&
      e.pub_year <= toYear
  );
}

// D3 graph construction
export const EMPTY_GRAPH: GraphData = {
  nodes: [],
  edges: [],
  maxWeight: 1,
  maxDegree: 1,
};

/**
 * Build D3-ready graph data from an AggregatedConcept.
 *
 * Cheap: no event scanning. Takes what it needs from the aggregation
 * and applies min-edge and max-node filters.
 *
 * GraphNode carries no provenance (doc_ids, years, scores).
 * For drill-down, callers query AggregatedConcept.byToken[node.id] directly.
 */
export function buildGraph(
  agg: AggregatedConcept,
  minEdgeWeight: number,
  maxNodes: number
): GraphData {
  // Collect all qualifying edges from the aggregation
  const filteredEdges: Array<[string, string, number]> = [];

  for (const [tokenA, stats] of agg.byToken) {
    for (const [tokenB, count] of stats.coOccurrences) {
      // Deduplicate: only emit A→B, not B→A (coOccurrences is symmetric)
      if (tokenA < tokenB && count >= minEdgeWeight) {
        filteredEdges.push([tokenA, tokenB, count]);
      }
    }
  }

  // Compute degree in filtered graph
  const degreeMap = new Map<string, number>();

  for (const [a, b] of filteredEdges) {
    degreeMap.set(a, (degreeMap.get(a) ?? 0) + 1);
    degreeMap.set(b, (degreeMap.get(b) ?? 0) + 1);
  }

  if (degreeMap.size === 0) return EMPTY_GRAPH;

  // Top-N by degree
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
