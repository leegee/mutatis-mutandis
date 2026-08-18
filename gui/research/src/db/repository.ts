import { getDatabase } from "./database";
import type { Entity, EntityType } from "~/domain/entity";
import type { Relation, RelationType } from "~/domain/relation";
import type { Evidence, ProjectMetadata, ResearchProject } from "~/domain/project";
import { validateProject, isValidProject, } from "~/domain/validateProject";

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
  value: unknown,
): Promise<void> {
  const validation = validateProject(value);

  if (!validation.valid) {
    throw new Error(
      validation.errors
        .map(
          (error) =>
            `${ error.path }: ${ error.message }`,
        )
        .join("\n"),
    );
  }

  if (!isValidProject(value)) {
    // Should never happen because validateProject()
    // has already succeeded.
    throw new Error("Invalid research project.");
  }

  const project = value;

  // project is now ResearchProject
  const db = getDatabase();

  await db.transaction(
    "rw",
    db.entities,
    db.relations,
    db.evidence,
    db.projectMetadata,
    async () => {
      // ...
    },
  );
}



export async function exportProject(): Promise<ResearchProject> {
  const db = getDatabase();

  const [
    metadataRecord,
    entities,
    relations,
    evidence,
  ] = await Promise.all([
    db.projectMetadata.get("project"),
    db.entities.toArray(),
    db.relations.toArray(),
    db.evidence.toArray(),
  ]);

  if (!metadataRecord) {
    throw new Error(
      "Project metadata has not been initialized.",
    );
  }

  const {
    id: _id,
    ...metadata
  } = metadataRecord;

  return {
    version: 1,
    metadata,
    entities,
    relations,
    evidence,
  };
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


export async function getProjectMetadata(): Promise<
  ProjectMetadata | undefined
> {
  const record =
    await getDatabase()
      .projectMetadata
      .get("project");

  if (!record) {
    return undefined;
  }

  const {
    id: _id,
    ...metadata
  } = record;

  return metadata;
}

export async function saveProjectMetadata(
  metadata: ProjectMetadata,
): Promise<void> {
  await getDatabase()
    .projectMetadata
    .put({
      id: "project",
      ...metadata,
    });
}

export async function listEvidence(): Promise<Evidence[]> {
  return getDatabase()
    .evidence
    .orderBy("createdAt")
    .toArray();
}

export async function createEvidence(
  evidence: Evidence,
): Promise<Evidence> {
  await getDatabase()
    .evidence
    .add(evidence);

  return evidence;
}


export async function addEntityTag(
  entity: Entity,
  tag: string,
): Promise<Entity> {
  const value = tag.trim();

  if (!value) {
    return entity;
  }

  const exists = entity.tags.some(
    (existing) => existing.toLocaleLowerCase() === value.toLocaleLowerCase()
  );

  if (exists) {
    return entity;
  }

  return updateEntity(entity, {
    tags: [
      ...entity.tags,
      value,
    ],
  });
}


export async function removeEntityTag(
  entity: Entity,
  tag: string,
): Promise<Entity> {
  const value = tag.trim().toLocaleLowerCase();

  return updateEntity(entity, {
    tags: entity.tags.filter(
      (existing) => existing.toLocaleLowerCase() !== value),
  });
}


// async function setEntityTags(
//   entity: Entity,
//   tags: string[],
// ): Promise<Entity> {
//   const seen = new Set<string>();
//   const normalized: string[] = [];

//   for (const tag of tags) {
//     const value = tag.trim();

//     if (!value) {
//       continue;
//     }

//     const key = value.toLocaleLowerCase();

//     if (seen.has(key)) {
//       continue;
//     }

//     seen.add(key);
//     normalized.push(value);
//   }

//   return updateEntity(entity, {
//     tags: normalized,
//   });
// }


export async function listTags(): Promise<string[]> {
  const entities = await getDatabase().entities.toArray();

  const tags = new Map<string, string>();

  for (const entity of entities) {
    for (const tag of entity.tags) {
      const value = tag.trim();

      if (!value) {
        continue;
      }

      const key = value.toLocaleLowerCase();

      if (!tags.has(key)) {
        tags.set(key, value);
      }
    }
  }

  return [...tags.values()].sort((a, b) => a.localeCompare(b),);
}
