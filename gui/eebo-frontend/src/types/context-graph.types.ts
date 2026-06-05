export type ViewMode = "aggregated" | "events";

export interface Neighbour {
  event_id: string;
  vector_id?: number;
  token: string;
  doc_id?: string;
  pub_year?: number;
  token_idx: string;
  window_id?: number;
  window_token_pos: number;
  score: number;
}

export interface Event {
  event_id: string; // The integers are too large for JS?
  vector_id: string;
  token: string;
  token_idx: number;
  doc_id: string;
  pub_year: number;
  window_id: number;
  window_token_pos: number;
}

export interface ConceptEvent extends Event {
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
