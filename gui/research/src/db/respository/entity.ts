// db/respository/entity

import type { Entity, EntityType } from "~/domain/entity";
import { getDatabase } from "../database";
import { id, now } from "./utils";

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
	await getDatabase().transaction(
		"rw",
		getDatabase().entities,
		getDatabase().relations,
		async () => {
			await getDatabase().entities.delete(entityId);

			await getDatabase().relations.where("sourceId").equals(entityId).delete();

			await getDatabase().relations.where("targetId").equals(entityId).delete();
		},
	);
}

export async function listEntities(): Promise<Entity[]> {
	return getDatabase().entities.orderBy("label").toArray();
}
