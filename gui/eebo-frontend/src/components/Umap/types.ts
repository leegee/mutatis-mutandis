export interface PointData {
  event_id: string;
  token_idx: number;
  doc_id: string;
  token: string;
  pub_year: number;
  // local projection
  nx: number;
  ny: number;
  // global projection
  gnx: number;
  gny: number;
  // any additional augmented fields the parent has attached
  [key: string]: unknown;
}

export interface ConceptDataset {
  concept: string;
  points: PointData[];
  bounds: Bounds;
  globalBounds: Bounds;
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
