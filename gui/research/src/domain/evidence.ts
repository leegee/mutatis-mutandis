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
