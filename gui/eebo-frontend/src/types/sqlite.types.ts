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
  lat: number;
  lng: number;
}

export interface SqliteEvent {
  concept?: string;
  doc_id: string;
  event_id: string; // The integers are too large for JS?
  geom?: string;    // Should drop this, SQLlite does not support geom ST_*
  lat: number;
  lng: number;
  pub_year: number;
  token: string;
  token_idx: number;
  vector_id: string;
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