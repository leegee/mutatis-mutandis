export type EntityType =
    | "concept"
    | "lexeme"
    | "motif"
    | "animal"
    | "person"
    | "source";

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
