import Dexie, { type Table } from "dexie";

import type { Entity } from "~/domain/entity";
import type { ProjectMetadata } from "~/domain/project";
import type { Relation } from "~/domain/relation";

interface ProjectMetadataRecord extends ProjectMetadata {
	id: "project";
}

class ResearchDatabase extends Dexie {
	entities!: Table<Entity, string>;
	relations!: Table<Relation, string>;
	projectMetadata!: Table<ProjectMetadataRecord, string>;

	constructor() {
		super("research-map");

		this.version(1).stores({
			entities: "id, type, label",
			relations: "id, sourceId, targetId, type",
		});

		this.version(2)
			.stores({
				entities: "id, type, label",
				relations: "id, sourceId, targetId, type",
				projectMetadata: "id",
			})
			.upgrade(async (tx) => {
				const metadata = await tx.table("projectMetadata").get("project");

				if (!metadata) {
					const timestamp = new Date().toISOString();

					await tx.table("projectMetadata").add({
						id: "project",
						title: "Research Map",
						description: "",
						createdAt: timestamp,
						updatedAt: timestamp,
					});
				}
			});
	}
}

let database: ResearchDatabase | undefined;

export function getDatabase(): ResearchDatabase {
	if (typeof window === "undefined") {
		throw new Error("Research database is only available in the browser");
	}

	database ??= new ResearchDatabase();

	return database;
}
