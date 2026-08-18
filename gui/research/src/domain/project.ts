import type { Entity } from "./entity";
import type { Relation } from "./relation";

export type EvidenceStatus =
  | "primary"
  | "secondary"
  | "interpretive"
  | "speculative";

export interface Evidence {
  id: string;

  sourceId: string;

  entityIds: string[];
  relationIds: string[];

  quote?: string;
  observation: string;

  status: EvidenceStatus;

  notes?: string;

  createdAt: string;
}

export interface ResearchProject {
  version: 1;

  metadata: {
    title: string;
    description: string;
    createdAt: string;
    updatedAt: string;
  };

  entities: Entity[];

  relations: Relation[];

  evidence: Evidence[];
}
