import { getEventsByIds } from "../../services/db/getEventsByIds";
import type { Event } from "../../types";
import type { ConceptDataset } from "./types";

export async function loadDatasets() {
    const [a, b] = await Promise.all([
        fetch("/umap/concept/LIBERTY.json").then((r) => r.json()),
        fetch("/umap/concept/PREROGATIVE.json").then((r) => r.json()),
    ]);

    // Augment each point with its concept name so colorBy:"concept" works.
    const augmented = (d: ConceptDataset) => ({
        ...d,
        points: d.points.map(
            (p) => ({ ...p, concept: d.concept })),
    });

    const datasetsTagged = [a, b].map(augmented);

    const eids = datasetsTagged.map(_ => _.points.map(_ => _.event_id));

    const events = await getEventsByIds(eids[0]);
    console.log("EVENTS-------------------\n", events);

    return datasetsTagged;
}