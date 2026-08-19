import type { Relation, RelationType } from "~/domain/relation";
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

export async function deleteRelation(relationId: string): Promise<void> {
	const db = getDatabase();

	await db.transaction("rw", db.relations, db.evidence, async () => {
		const evidence = await db.evidence.toArray();

		for (const item of evidence) {
			if (!item.relationIds.includes(relationId)) {
				continue;
			}

			await db.evidence.put({
				...item,
				relationIds: item.relationIds.filter((id) => id !== relationId),
			});
		}

		await db.relations.delete(relationId);
	});
}
