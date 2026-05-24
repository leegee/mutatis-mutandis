/**
 * ConceptGraph.types.ts
 *
 * Shared types for the ConceptGraph pipeline.
 * Separated so they can be imported by data layer, graph layer,
 * and UI independently.
 */

// Raw Tier2 data shape (as received from API)
export interface Neighbour {
  token: string;
  score: number;
  event_id?: number;
  doc_id?: string;
  window_id?: number;
}

export interface ConceptEvent {
  event_id?: number;
  token?: string;
  doc_id?: string;
  pub_year?: number;
  slice_start?: number;
  slice_end?: number;
  neighbours: Neighbour[];
}

export interface ConceptData {
  n_events: number;
  events: ConceptEvent[];
}

export interface Tier2Data {
  [concept: string]: ConceptData;
}

// Intermediate aggregation layer
//
// Produced once per concept by aggregateConcept().
// Carries full provenance so downstream consumers (graph builder,
// document drill-down, temporal split) can each take what they need
// without re-scanning the raw events.

export interface TokenStats {
  token: string;
  // co-occurrence counts with other tokens in neighbourhood space
  coOccurrences: Map<string, number>;
  // documents in which this token appeared in the concept's neighbourhood
  docs: Set<string>;
  // total neighbourhood appearances (across all concept events)
  totalAppearances: number;
}

export interface AggregatedConcept {
  // keyed by token string
  byToken: Map<string, TokenStats>;
  nEvents: number;
}

// Graph layer (D3-facing)
//
// Produced by buildGraph() from an AggregatedConcept.
// Carries no provenance — provenance lives in AggregatedConcept.

import type * as d3 from "d3";

export interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  degree: number;
}

export interface GraphEdge extends d3.SimulationLinkDatum<GraphNode> {
  source: GraphNode;
  target: GraphNode;
  weight: number;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  maxWeight: number;
  maxDegree: number;
}