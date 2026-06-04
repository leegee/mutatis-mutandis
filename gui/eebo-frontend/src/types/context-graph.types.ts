export type ViewMode = "aggregated" | "events";

export interface Neighbour {
  event_id?: number;
  vector_id?: number;
  token: string;
  doc_id?: string;
  pub_year?: number;
  token_idx: string;
  window_id?: number;
  window_token_pos: number;
  score: number;
}

export interface ConceptEvent {
  event_id: number;
  vector_id: number;
  token: string;
  token_idx: number;
  doc_id: string;
  pub_year: number;
  window_id: number;
  window_token_pos: number;
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

export interface YearBucket {
  year: number;
  count: number;
}
