import type { Entity } from "./entity";
import type { Evidence } from "./evidence";
import type { Relation } from "./relation";

export interface ProjectMetadata {
	title: string;
	description: string;
	createdAt: string;
	updatedAt: string;
}

export interface ResearchProject {
	version: 1;
	metadata: ProjectMetadata;
	entities: Entity[];
	relations: Relation[];
	evidence: Evidence[];
}
