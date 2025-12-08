import { Database } from "bun:sqlite";

export interface EeboRow {
    id?: number;
    idno: string;
    xml_created: number | null;
}

export class EeboDAO {
    private db: Database;

    constructor(dbFile: string) {
        this.db = new Database(dbFile);
        this.ensureColumn();
    }

    /** Ensure xml_created column exists (boolean/null) */
    private ensureColumn() {
        const col = this.db
            .prepare("PRAGMA table_info(eebo)")
            .all()
            .map((r: any) => r.name)
            .includes("xml_created");

        if (!col) {
            console.log("Adding xml_created column to eebo table...");
            this.db.run("ALTER TABLE eebo ADD COLUMN xml_created INTEGER DEFAULT NULL");
        }
    }

    /** Get xml_created for a given idno */
    isProcessed(idno: string): boolean {
        const row = this.db
            .prepare("SELECT xml_created FROM eebo WHERE idno = ?")
            .get(idno) as EeboRow | undefined;
        return row?.xml_created === 1;
    }

    /** Update xml_created status (1 = success, 0 = failed) */
    updateStatus(idno: string, status: 0 | 1) {
        this.db.prepare("UPDATE eebo SET xml_created = ? WHERE idno = ?").run(status, idno);
    }

    /** Optional: insert new record if needed */
    insert(row: EeboRow) {
        this.db
            .prepare(
                "INSERT INTO eebo (idno, xml_created) VALUES (?, ?)"
            )
            .run(row.idno, row.xml_created ?? null);
    }
}
