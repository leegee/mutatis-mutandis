export interface Neighbour {
  token: string; score: number;
  event_id?: number; doc_id?: string; pub_year?: number; window_id?: number;
}

export interface ConceptEvent {
  event_id?: number; token?: string; doc_id?: string; pub_year?: number;
  neighbours: Neighbour[];
}

export interface ConceptData {
  n_events: number; year_min?: number; year_max?: number; events: ConceptEvent[];
}

export interface Tier2Data { [concept: string]: ConceptData; }

export interface TokenBin {
  token: string; eventCount: number;
  neighbourFreq: Map<string, number>; neighbourScoreSum: Map<string, number>;
  topNeighbours: Array<{ token: string; freq: number; meanScore: number }>;
  docs: Map<string, number | undefined>; years: Set<number>;
}

export interface ContextNode {
  id: string; kind: "hub" | "neighbour" | "event";
  eventCount: number; hubDegree: number; degree: number;
  token?: string; doc_id?: string; pub_year?: number;
}

export interface HubHubEdge { kind: "hub-hub"; sourceId: string; targetId: string; weight: number; }

export interface HubNbEdge { kind: "hub-neighbour"; sourceId: string; targetId: string; weight: number; }

export type AnyEdge = HubHubEdge | HubNbEdge;

export interface ContextGraphData {
  nodes: ContextNode[]; hubHubEdges: HubHubEdge[]; hubNbEdges: HubNbEdge[];
  allEdges: AnyEdge[]; maxHubHubWeight: number; maxEventCount: number; maxHubDegree: number;
}