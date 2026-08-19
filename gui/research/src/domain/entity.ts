export const entityTypes = [
	"concept",
	"lexeme",
	"motif",
	"animal",
	"person",
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
