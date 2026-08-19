import type { Entity } from "~/domain/entity";
import { getDatabase } from "../database";
import { updateEntity } from ".";

export async function addEntityAlias(
	entity: Entity,
	alias: string,
): Promise<Entity> {
	const value = alias.trim();

	if (!value) {
		return entity;
	}

	const exists = entity.aliases.some(
		(existing) => existing.toLocaleLowerCase() === value.toLocaleLowerCase(),
	);

	if (exists) {
		return entity;
	}

	return updateEntity(entity, {
		aliases: [...entity.aliases, value],
	});
}

export async function removeEntityAlias(
	entity: Entity,
	alias: string,
): Promise<Entity> {
	const value = alias.trim().toLocaleLowerCase();

	return updateEntity(entity, {
		aliases: entity.aliases.filter(
			(existing) => existing.toLocaleLowerCase() !== value,
		),
	});
}

export async function listAliases(): Promise<string[]> {
	const entities = await getDatabase().entities.toArray();
	const aliases = new Map<string, string>();

	for (const entity of entities) {
		for (const alias of entity.aliases) {
			const value = alias.trim();
			if (!value) {
				continue;
			}

			const key = value.toLocaleLowerCase();
			if (!aliases.has(key)) {
				aliases.set(key, value);
			}
		}
	}

	return [...aliases.values()].sort((a, b) => a.localeCompare(b));
}
