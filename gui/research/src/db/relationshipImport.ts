// db/respository/relationshipImport.ts

import type { Entity } from "~/domain/entity";
import type { Relation, RelationType } from "~/domain/relation";
import { getDatabase } from "./database";
import { id, now } from "./respository/utils";

export interface RelationshipImport {
	source: string;
	type: RelationType;
	target: string;
}

export interface RelationshipImportResult {
	createdEntities: Entity[];
	createdRelations: Relation[];
}

function normalise(value: string): string {
	return value.trim();
}

function findEntity(
	entities: Entity[],
	label: string,
): Entity | undefined {
	const wanted = normalise(label);

	return entities.find(
		(entity) =>
			entity.label === wanted ||
			entity.aliases.includes(wanted),
	);
}

export async function importRelationships(
	imports: RelationshipImport[],
): Promise<RelationshipImportResult> {
	const db = getDatabase();

	return db.transaction(
		"rw",
		db.entities,
		db.relations,
		async () => {
			const entities = await db.entities.toArray();
			const relations = await db.relations.toArray();

			const createdEntities: Entity[] = [];
			const createdRelations: Relation[] = [];

			// Keep entities created during this transaction in the same
			// lookup collection so repeated new entities are not duplicated.
			const workingEntities = [...entities];

			for (const item of imports) {
				let source = findEntity(workingEntities, item.source);

				if (!source) {
					const timestamp = now();

					source = {
						id: id(),
						label: normalise(item.source),
						aliases: [],
						type: "concept",
						description: "",
						tags: [],
						createdAt: timestamp,
						updatedAt: timestamp,
					};

					await db.entities.add(source);
					workingEntities.push(source);
					createdEntities.push(source);
				}

				let target = findEntity(workingEntities, item.target);

				if (!target) {
					const timestamp = now();

					target = {
						id: id(),
						label: normalise(item.target),
						aliases: [],
						type: "concept",
						description: "",
						tags: [],
						createdAt: timestamp,
						updatedAt: timestamp,
					};

					await db.entities.add(target);
					workingEntities.push(target);
					createdEntities.push(target);
				}

				const alreadyExists =
					relations.some(
						(relation) =>
							relation.sourceId === source.id &&
							relation.type === item.type &&
							relation.targetId === target.id,
					) ||
					createdRelations.some(
						(relation) =>
							relation.sourceId === source.id &&
							relation.type === item.type &&
							relation.targetId === target.id,
					);

				if (alreadyExists) {
					continue;
				}

				const relation: Relation = {
					id: id(),
					sourceId: source.id,
					type: item.type,
					targetId: target.id,
					createdAt: now(),
				};

				await db.relations.add(relation);
				createdRelations.push(relation);
			}

			return {
				createdEntities,
				createdRelations,
			};
		},
	);
}

export interface RelationshipImportResolution {
	source: Entity | undefined;
	target: Entity | undefined;
	relationExists: boolean;
}

export async function resolveRelationshipImport(
	item: RelationshipImport,
): Promise<RelationshipImportResolution> {
	const db = getDatabase();

	const entities = await db.entities.toArray();

	const source = findEntity(entities, item.source);
	const target = findEntity(entities, item.target);

	if (!source || !target) {
		return {
			source,
			target,
			relationExists: false,
		};
	}

	const relationExists =
		(await db.relations
			.where("sourceId")
			.equals(source.id)
			.toArray())
			.some(
				(relation) =>
					relation.type === item.type &&
					relation.targetId === target.id,
			);

	return {
		source,
		target,
		relationExists,
	};
}

