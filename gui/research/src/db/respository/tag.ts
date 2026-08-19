import type { Entity } from "~/domain/entity";
import { getDatabase } from "../database";
import { updateEntity } from "./entity";

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
