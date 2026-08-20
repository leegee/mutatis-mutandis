export const relationTypes = [
     "derived-from",
     "parallels", 
    "associated-with",
    "attested-in",
    "cognate-of",
    "contrasts-with",
    "describes",
    "exhibits",
    "expresses",
    "possibly-derived-from",
    "related-to",
    "supports",
    "variant-of",
] as const;

export type RelationType = (typeof relationTypes)[number];

export interface Relation {
	id: string;
	sourceId: string;
	type: RelationType;
	targetId: string;
	createdAt: string;
}
