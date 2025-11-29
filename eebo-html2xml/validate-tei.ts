import { spawnSync } from "bun";
import path from "path";

export function validateTEI(filePath: string) {
    const dtdPath = path.resolve("./tei-dtd/tei_all.dtd");

    const result = spawnSync({
        cmd: ["xmllint", "--noout", "--dtdvalid", dtdPath, filePath]
    });

    if (result.exitCode !== 0) {
        console.error(`❌ TEI validation failed for ${filePath}`);
        console.error(result.stderr.toString());
        return false;
    }

    console.log(`✔ TEI valid: ${filePath}`);
    return true;
}
