import { RelationType, Relation } from "~/domain/relation";
import { getDatabase } from "../database";
import { id, now } from "./utils";

export async function createRelation(
  sourceId: string,
  type: RelationType,
  targetId: string,
): Promise<Relation> {
  const relation: Relation = {
    id: id(),
    sourceId,
    type,
    targetId,
    createdAt: now(),
  };

  await getDatabase().relations.add(relation);
  return relation;
}

export async function listRelations(): Promise<Relation[]> {
  return getDatabase().relations.toArray();
}


export async function updateRelation(
  relation: Relation,
  changes: Partial<Omit<Relation, "id" | "createdAt">>,
): Promise<Relation> {
  const updated: Relation = {
    ...relation,
    ...changes,
  };

  await getDatabase().relations.put(updated);
  return updated;
}

export async function deleteRelation(
  relationId: string,
): Promise<void> {
  await getDatabase().relations.delete(relationId);
}

