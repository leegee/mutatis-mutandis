import { getEventsByIds } from "../../services/db/getEventsByIds";
import type { Event } from "../../types";
import type { ConceptDataset } from "./types";

export async function loadDatasets() {
    const [a, b] = await Promise.all([
        fetch("/umap/concept/PREROGATIVE.json").then((r) => r.json()),
        fetch("/umap/concept/CHURCH.json").then((r) => r.json()),
    ]);
    console.log("[Umap.index] loaded", a.points.map((_: Event) => _.event_id))

    // Augment each point with its concept name so colorBy:"concept" works.
    const augmented = (d: ConceptDataset) => ({
        ...d,
        points: d.points.map((p) => ({ ...p, concept: d.concept, doc_id: 'a', token: d.concept })),
    });

    const datasetsTagged = [a, b].map(augmented);
    console.log("[Umap.index] augmented");

    const eids = datasetsTagged.map(_ => _.points.map(_ => _.event_id));
    console.log("EIDS-------------------\n", eids[0]);

    const events = await getEventsByIds(eids[0]);
    console.log("EVENTS-------------------\n", events);

    return datasetsTagged;
}