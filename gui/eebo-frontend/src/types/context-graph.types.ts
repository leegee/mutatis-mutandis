import type { Event, Neighbour } from "./sqlite";


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
