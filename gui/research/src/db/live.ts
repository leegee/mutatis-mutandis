import { liveQuery } from "dexie";
import { getDatabase } from "./database";

export function liveEntities() {
	return liveQuery(() => getDatabase().entities.orderBy("label").toArray());
}

export function liveRelations() {
	return liveQuery(() => getDatabase().relations.toArray());
}

