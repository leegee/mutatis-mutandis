import type { SqliteEvent } from '../../types'

export interface PointData extends Omit<SqliteEvent, "vector_id"> {
  event_id: string;
  token_idx: number;
  // local projection
  nx: number;
  ny: number;
  // global projection
  gnx: number;
  gny: number;
  concept: string;
  // any additional augmented fields the parent has attached
  [key: string]: unknown;

  // Clustering
  cluster_id?: number;
  cluster_label?: string;
  umap_x?: number;
  umap_y?: number;
}

export interface ConceptDataset {
  concept: string;
  origin?: string | undefined;
  points: PointData[];
  bounds: Bounds;
  globalBounds: Bounds;
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
