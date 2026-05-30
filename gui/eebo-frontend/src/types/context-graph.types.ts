export type ViewMode = "aggregated" | "events";

export interface Neighbour {
  token: string;
  score: number;
  event_id?: number;
  doc_id?: string;
  pub_year?: number;
  window_id?: number;
}

export interface ConceptEvent {
  event_id?: number;
  vector_id?: number;

  token?: string;
  token_idx?: number;

  doc_id?: string;
  pub_year?: number;

  window_id?: number;
  window_token_pos?: number;

  neighbours: Neighbour[];
}

export interface ConceptData {
  n_events: number;
  year_min?: number;
  year_max?: number;
  events: ConceptEvent[];
}

export interface Tier2Data {
  [concept: string]: ConceptData;
}

export interface TokenBin {
  token: string;
  eventCount: number;
  neighbourFreq: Map<string, number>;
  /**
   * Accumulated raw scores per neighbour token, tracked separately from
   * frequency so that meanScore = scoreSum / freq is exact and not
   * occurrence-weighted.
   */
  neighbourScoreSum: Map<string, number>;
  topNeighbours: Array<{ token: string; freq: number; meanScore: number }>;
  docs: Map<string, number | undefined>;
  years: Set<number>;
}

/** Unified simulation node for hubs, events, and neighbours. */
export interface ContextNode extends d3.SimulationNodeDatum {
  id: string;
  kind: "hub" | "neighbour" | "event";
  /** Hub only: number of corpus events aggregated here. */
  eventCount: number;
  /** Hub only: edge count from hub-hub edges. */
  hubDegree: number;
  /** Both: total edge count (hub-hub + hub-neighbour spokes). */
  degree: number;
  /** Event node only. */
  token?: string;
  doc_id?: string;
  pub_year?: number;
}

/** Hub ↔ Hub edge: cosine similarity between neighbour-freq vectors. */
export interface HubHubEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-hub";
  source: ContextNode;
  target: ContextNode;
  weight: number;
}

/** Hub/Event > Neighbour spoke: normalised frequency or raw score. */
export interface HubNbEdge extends d3.SimulationLinkDatum<ContextNode> {
  kind: "hub-neighbour";
  source: ContextNode;
  target: ContextNode;
  weight: number;
}

export type AnyEdge = HubHubEdge | HubNbEdge;

export interface ContextGraphData {
  nodes: ContextNode[];
  hubHubEdges: HubHubEdge[];
  hubNbEdges: HubNbEdge[];
  allEdges: AnyEdge[];
  maxHubHubWeight: number;
  maxEventCount: number;
  maxHubDegree: number;
}
