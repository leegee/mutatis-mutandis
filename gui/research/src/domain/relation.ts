export type RelationType =
  | "related-to"
  | "contrasts-with"
  | "describes"
  | "expresses"
  | "associated-with"
  | "cognate-of"
  | "variant-of"
  | "attested-in"
  | "supports"
  | "possibly-derived-from";

export interface Relation {
  id: string;
  sourceId: string;
  type: RelationType;
  targetId: string;
  createdAt: string;
}