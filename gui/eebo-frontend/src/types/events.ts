// src/types/events.ts

export type SemanticNeighbour = {
  vector_id: number;
  token: string;
  similarity: number;
};

export type SemanticEvent = {
  id: string;

  vector_id: number;
  token: string;
  doc_id: string;
  filepath: string;

  concept: string;
  slice: string;

  x: number;
  y: number;

  neighbours: SemanticNeighbour[];
};