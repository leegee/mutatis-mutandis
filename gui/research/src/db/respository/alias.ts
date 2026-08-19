import type { Entity } from "~/domain/entity";

import { updateEntity, } from ".";

export async function addEntityAlias(
  entity: Entity,
  alias: string,
): Promise<Entity> {
  const value = alias.trim();

  if (!value) {
    return entity;
  }

  const exists = entity.aliases.some(
    (existing) =>
      existing.toLocaleLowerCase() ===
      value.toLocaleLowerCase(),
  );

  if (exists) {
    return entity;
  }

  return updateEntity(entity, {
    aliases: [
      ...entity.aliases,
      value,
    ],
  });
}

export async function removeEntityAlias(
  entity: Entity,
  alias: string,
): Promise<Entity> {
  const value = alias
    .trim()
    .toLocaleLowerCase();

  return updateEntity(entity, {
    aliases: entity.aliases.filter(
      (existing) =>
        existing.toLocaleLowerCase() !== value,
    ),
  });
}
