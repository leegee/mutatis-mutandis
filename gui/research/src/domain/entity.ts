// domain/entity.ts

export const entityTypes = [
	"animal",
	"concept",
	"evidence",
	"group",
	"lexeme",
	"motif",
	"person",
	"quote",
	"source",
] as const;

export type EntityType = (typeof entityTypes)[number];

export interface Entity {
	id: string;
	type: EntityType;
	label: string;
	aliases: string[];
	description?: string;
	tags: string[];
	createdAt: string;
	updatedAt: string;
}
