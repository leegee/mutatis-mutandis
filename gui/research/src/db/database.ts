import Dexie, { type Table } from "dexie";
import type { Entity } from "~/domain/entity";
import type { Relation } from "~/domain/relation";

class ResearchDatabase extends Dexie {
    entities!: Table<Entity, string>;
    relations!: Table<Relation, string>;

    constructor() {
        super("research-map");

        this.version(1).stores({
            entities: "id, type, label",
            relations: "id, sourceId, targetId, type",
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
