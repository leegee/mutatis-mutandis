import fs from "fs";
import { Database } from "bun:sqlite";
import { htmlDir, htmlFile, jsonFile, normalisePhiloId } from "./lib";
const dbFile = "./eebo-tcp_metadata.sqlite";

if (!fs.existsSync(htmlDir)) fs.mkdirSync(htmlDir, { recursive: true });

const db = new Database(dbFile);

async function fetchJSON(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.error("Fetch error:", url, err);
        return null;
    }
}

async function getNumericId(permalink) {
    if (!permalink) return null;
    try {
        const res = await fetch(permalink, { redirect: "follow" });
        const finalUrl = res.url;
        const match = finalUrl.match(/\/navigate\/(\d+)\/table-of-contents/);
        if (!match) return null;
        return match[1]; // e.g., "17338"
    } catch (err) {
        console.error("Error following permalink:", permalink, err);
        return null;
    }
}

async function resolvePhiloDiv1Id(philoDocId) {
    if (!philoDocId) return null;

    const url = `https://artflsrv04.uchicago.edu/philologic4.7/eebo_08_2022/scripts/get_table_of_contents.py?philo_id=${encodeURIComponent(philoDocId)}+1`;
    const data = await fetchJSON(url);
    if (!data?.toc?.length) return null;

    const div1 = data.toc.find(d => d.philo_type === "div1");
    if (!div1) return null;

    return div1.philo_id; // e.g., "17338 1"
}

// --- Download EEBO HTML + metadata ---
async function downloadEEBO(philoDiv1Id) {
    if (!philoDiv1Id) return;

    const url = `https://artflsrv04.uchicago.edu/philologic4.7/eebo_08_2022/reports/navigation.py?&philo_id=${encodeURIComponent(philoDiv1Id)}`;
    const data = await fetchJSON(url);
    if (!data) return;

    const text = data.text || "";
    const metadata = data.metadata_fields || {};

    if (!text) {
        console.warn("No text for", philoDiv1Id);
        return;
    }

    fs.writeFileSync(htmlFile(philoDiv1Id), text, "utf-8");
    fs.writeFileSync(jsonFile(philoDiv1Id), JSON.stringify(metadata, null, 2), "utf-8");

    console.log(`Saved ${htmlFile(philoDiv1Id)}`);
    console.log(`Saved ${jsonFile(philoDiv1Id)}`);
    console.log(`Saved "${metadata.title ? metadata.title.substring(0, 15) + '...' : philoDiv1Id}" → HTML + JSON`);
    console.log("\n");
}


async function main() {
    const rows = db.query("SELECT id, permalink, philo_div1_id FROM eebo");

    for (const row of rows) {
        const { id, permalink, philo_div1_id } = row;

        let philoDiv1Id = philo_div1_id;

        if (!philoDiv1Id) {
            const philoDocId = await getNumericId(permalink);
            if (!philoDocId) {
                console.warn("Skipping row", id, "cannot resolve numeric philo_doc_id");
                continue;
            }

            philoDiv1Id = await resolvePhiloDiv1Id(philoDocId);
            if (!philoDiv1Id) {
                console.warn("Skipping row", id, "cannot resolve philo_div1_id");
                continue;
            }

            db.run("UPDATE eebo SET philo_div1_id = ? WHERE id = ?", philoDiv1Id, id);
            console.log(`Resolved row ${id} → philo_div1_id = ${philoDiv1Id}`);
        }

        if (fs.existsSync(htmlFile(philoDiv1Id)) && fs.existsSync(jsonFile(philoDiv1Id))) {
            console.log(`${htmlFile(philoDiv1Id)} exists.`)
        } else {
            await downloadEEBO(philoDiv1Id);
        }
    }

    console.log("Done.");
}

main();
