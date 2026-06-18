import type { YearBucket } from "../../types";

export type NodeKind = 0 | 1 | 2;

export interface NodeMeta {
  id: string;       // stable string id used for index lookups
  kind: NodeKind;
  label: string;
  docId?: string;
  pubYear?: number | null;
  windowId?: number | null;
  degree?: number | null;
  tokenIdx: number;
}

export type EdgeKind = 0 | 1 | 2;

export interface EdgeMeta {
  srcIdx: number; // index into nodes array
  kind: EdgeKind;
  tgtIdx: number;
  weight: number; // 0–1
}

export type Graph2Data = {
  nodes: NodeMeta[];
  edges: EdgeMeta[];
  years?: YearBucket[];
};

export const NODE_KIND = {
  EVENT: 0,
  NEIGHBOUR: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;

export const EDGE_KIND = {
  SEMANTIC: 0,
  COWINDOW: 1,
  CONCEPT: 2,
} as const satisfies Record<string, NodeKind>;
