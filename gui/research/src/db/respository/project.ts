import type { ProjectMetadata, ResearchProject } from "~/domain/project";
import { isValidProject, validateProject } from "~/domain/validateProject";
import { getDatabase } from "../database";
import { now } from "./utils";

export async function importProject(value: unknown): Promise<void> {
	const validation = validateProject(value);

	if (!validation.valid) {
		throw new Error(validation.errors.map((error) => `${error.path}: ${error.message}`).join("\n"));
	}

	if (!isValidProject(value)) {
		throw new Error("Invalid research project.");
	}

	const project = value as ResearchProject;
	const db = getDatabase();

	await db.transaction("rw", db.entities, db.relations, db.projectMetadata, async () => {
		/*
		 * Import replaces the current project.
		 */
		await db.entities.clear();
		await db.relations.clear();
		await db.projectMetadata.clear();

		await db.entities.bulkAdd(project.entities);
		await db.relations.bulkAdd(project.relations);

		await db.projectMetadata.put({
			id: "project",
			...project.metadata,
		});
	});
}

export async function exportProject(): Promise<ResearchProject> {
	const db = getDatabase();

	const [metadataRecord, entities, relations] = await Promise.all([
		db.projectMetadata.get("project"),
		db.entities.toArray(),
		db.relations.toArray(),
	]);

	if (!metadataRecord) {
		throw new Error("Project metadata has not been initialized.");
	}

	const { id: _id, ...metadata } = metadataRecord;

	return {
		version: 1,
		metadata,
		entities,
		relations,
	};
}

export async function getProjectMetadata(): Promise<ProjectMetadata | undefined> {
	const record = await getDatabase().projectMetadata.get("project");

	if (!record) {
		return undefined;
	}

	const { id: _id, ...metadata } = record;

	return metadata;
}

export async function saveProjectMetadata(metadata: ProjectMetadata): Promise<void> {
	await getDatabase().projectMetadata.put({
		id: "project",
		...metadata,
	});
}

export async function resetProject(): Promise<void> {
	const db = getDatabase();

	await db.transaction("rw", db.entities, db.relations, db.projectMetadata, async () => {
		await db.entities.clear();
		await db.relations.clear();

		const metadata = await db.projectMetadata.get("project");

		if (metadata) {
			await db.projectMetadata.put({
				...metadata,
				title: "Research Map",
				description: "",
				updatedAt: now(),
			});
		}
	});
}
