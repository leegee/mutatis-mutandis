import type { EnrichedEvent } from '../../lib/eventExport';
import type { SqliteEvent } from '../../types'

export interface PointData extends EnrichedEvent, Omit<SqliteEvent, "vector_id"> {
  event_id: string;
  depth?: number;
  token_idx: number;
  // local projection
  nx: number;
  ny: number;
  // global projection
  gnx: number;
  gny: number;
  concept: string;

  // [key: string]: unknown;
  origin?: "concept" | "neighbours" | "concept_clusters";
  cluster_id?: number;
  cluster_label?: string;
}


export interface LabelPoint {
  id: string;
  cluster_label: string; // was title
  description: string;
  nx: number;
  ny: number;
  gnx: number;
  gny: number;
  clusterId?: string;
  type: "cluster_summary" | "keyword" | "note";
}

export interface LabelDataset {
  labels: LabelPoint[];
  minCentroidDistance: number;
}


export interface ConceptDatasetJSON {
  concept: string;
  origin?: string | undefined;
  points: PointData[];
  bounds: Bounds;
  globalBounds: Bounds;
}

export interface ConceptDatasetSqlite {
  concept: string;
  origin?: string | undefined;
  points: PointData[];
  bounds: Bounds;
  globalBounds: Bounds;
  type: string;
}


export interface BfsDataset {
  points: any[];
  type: "bfs_global";
  bounds: any;
  globalBounds: any;
  depth: number;
  k: number;
}

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export interface ViewBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  zoom: number;
}


