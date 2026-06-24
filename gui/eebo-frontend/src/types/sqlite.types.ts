export interface SqliteNeighbour {
  event_id: string;
  vector_id?: string;
  token: string;
  doc_id?: string;
  pub_year?: number;
  token_idx: string;
  window_id?: number;
  window_token_pos: number;
  score: number;
}

export interface SqliteEvent {
  event_id: string; // The integers are too large for JS?
  vector_id: string;
  token: string;
  token_idx: number;
  doc_id: string;
  pub_year: number;
  window_id: number;
  window_token_pos: number;
}

export interface SqliteEventWithNeighbours extends SqliteEvent {
  neighbours: SqliteNeighbour[];
}


export type EventQuery = {
  concept?: string;
  fromYear?: number;
  toYear?: number;
  selectedEventIds?: string[] | null;
  bbox?: [number, number, number, number];
};