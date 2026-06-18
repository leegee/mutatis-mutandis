import type { SqliteEventWithNeighbours, SqliteEvent, SqliteNeighbour } from "./sqlite.types";


export interface ConceptData {
  n_events: number;
  // year_min?: number;
  // year_max?: number;
  events: SqliteEventWithNeighbours[];
}

export interface Tier2Data {
  [concept: string]: ConceptData;
}

export interface YearBucket {
  year: number;
  count: number;
}

















