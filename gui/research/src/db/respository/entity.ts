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

export async function listEntities(): Promise<Entity[]> {
	return getDatabase().entities.orderBy("label").toArray();
}

export async function deleteEntity(entityId: string): Promise<void> {
	const db = getDatabase();

	await db.transaction(
		"rw",
		db.entities,
		db.relations,
		db.evidence,
		async () => {
			// An entity used as the source of evidence cannot be deleted.
			const sourceEvidenceCount = await db.evidence
				.where("sourceId")
				.equals(entityId)
				.count();

			if (sourceEvidenceCount > 0) {
				throw new Error(
					"This entity is used as the source of evidence and cannot be deleted.",
				);
			}

			// Remove references to the entity from evidence.
			const evidence = await db.evidence.toArray();

			for (const item of evidence) {
				if (!item.entityIds.includes(entityId)) {
					continue;
				}

				await db.evidence.put({
					...item,
					entityIds: item.entityIds.filter((id) => id !== entityId),
				});
			}

			// Delete the entity itself.
			await db.entities.delete(entityId);

			// Delete relationships involving the entity.
			await db.relations.where("sourceId").equals(entityId).delete();

			await db.relations.where("targetId").equals(entityId).delete();
		},
	);
}
