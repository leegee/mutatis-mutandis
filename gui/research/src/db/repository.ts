import { getDatabase } from "./database";
import type { Entity, EntityType } from "~/domain/entity";
import type { Relation, RelationType } from "~/domain/relation";
import type { ResearchProject } from "~/domain/project";

function id(): string {
  return crypto.randomUUID();
}

function now(): string {
  return new Date().toISOString();
}

export async function createEntity(
  label: string,
  type: EntityType = "concept",
  description = "",
  tags: string[] = [],
): Promise<Entity> {
  const timestamp = now();

  const entity: Entity = {
    id: id(),
    label,
    aliases: [],
    type,
    description,
    tags,
    createdAt: timestamp,
    updatedAt: timestamp,
  };

  await getDatabase().entities.add(entity);
  return entity;
}

export async function updateEntity(
  entity: Entity,
  changes: Partial<Omit<Entity, "id" | "createdAt">>,
): Promise<Entity> {
  const updated: Entity = {
    ...entity,
    ...changes,
    updatedAt: now(),
  };

  await getDatabase().entities.put(updated);
  return updated;
}

export async function deleteEntity(entityId: string): Promise<void> {
  await getDatabase().transaction("rw", getDatabase().entities, getDatabase().relations, async () => {
    await getDatabase().entities.delete(entityId);

    await getDatabase().relations
      .where("sourceId")
      .equals(entityId)
      .delete();

    await getDatabase().relations
      .where("targetId")
      .equals(entityId)
      .delete();
  });
}

export async function listEntities(): Promise<Entity[]> {
  return getDatabase().entities.orderBy("label").toArray();
}

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

export async function importProject(
  project: ResearchProject,
): Promise<void> {
  const db = getDatabase();

  if (project.version !== 1) {
    throw new Error(
      `Unsupported project version: ${ project.version }`,
    );
  }

  await db.transaction(
    "rw",
    db.entities,
    db.relations,
    async () => {
      await db.entities.clear();
      await db.relations.clear();

      if (project.entities.length > 0) {
        await db.entities.bulkAdd(project.entities);
      }

      if (project.relations.length > 0) {
        await db.relations.bulkAdd(project.relations);
      }
    },
  );
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

