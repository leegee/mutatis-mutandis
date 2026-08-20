import type { Entity } from "~/domain/entity";
import type { Relation, RelationType } from "~/domain/relation";

import { getDatabase } from "../database";
import { id, now } from "./utils";

export interface RelationshipImport {
	source: string;
	type: RelationType;
	target: string;
}

export interface RelationshipImportResult {
	createdEntities: Entity[];
	createdRelations: Relation[];
}

function findEntity(
	entities: Entity[],
	label: string,
): Entity | undefined {
	const value = label.trim();

	return entities.find(
		(entity) =>
			entity.label === value ||
			entity.aliases.includes(value),
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

			const workingEntities = [...entities];

			const createdEntities: Entity[] = [];
			const createdRelations: Relation[] = [];

			for (const item of imports) {
				let source = findEntity(
					workingEntities,
					item.source,
				);

				if (!source) {
					const timestamp = now();

					source = {
						id: id(),
						label: item.source.trim(),
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

				let target = findEntity(
					workingEntities,
					item.target,
				);

				if (!target) {
					const timestamp = now();

					target = {
						id: id(),
						label: item.target.trim(),
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

				const relationExists =
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

				if (relationExists) {
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
