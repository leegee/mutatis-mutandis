import * as path from 'node:path';
import * as fs from 'node:fs';
import { Database } from "bun:sqlite";

export interface EeboRow {
    id: string;
    xml_created: number | null;
}

export const DB_PATH = path.resolve(import.meta.dir, "../eebo-data/eebo-tcp_metadata.sqlite");

export class EeboDAO {
    private db: Database;

    constructor(dbFile?: string) {
        const dbPath = dbFile || DB_PATH;

        if (!fs.existsSync(dbPath)) throw new Error(`Cannot find dbFile "${dbPath}".`);

        this.db = new Database(dbPath);
        this.ensureColumn();
    }

    private ensureColumn() {
        // const col = this.db
        //     .prepare("PRAGMA table_info(eebo)")
        //     .all()
        //     .map((r: any) => r.name)
        //     .includes("xml_created");

        // if (!col) {
        //     console.log("Adding xml_created column to eebo table...");
        //     this.db.run("ALTER TABLE eebo ADD COLUMN xml_created INTEGER DEFAULT NULL");
        // }
    }

    /** Get xml_created for a given id */
    xmlGenerated(id: string): boolean {
        const row = this.db
            .prepare("SELECT xml_created FROM eebo WHERE id = ?")
            .get(id) as EeboRow | undefined;
        return row?.xml_created === 1;
    }

    /** Update xml_created status (1 = success, 0 = failed) */
    updateStatus(id: string, status: 0 | 1) {
        this.db.prepare("UPDATE eebo SET xml_created = ? WHERE id = ?").run(status, id);
    }

    /** Optional: insert new record if needed */
    insert(row: EeboRow) {
        this.db
            .prepare(
                "INSERT INTO eebo (id, xml_created) VALUES (?, ?)"
            )
            .run(row.id, row.xml_created ?? null);
    }
}
