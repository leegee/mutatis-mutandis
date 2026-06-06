import { getEventsByIds } from "../../services/db/getEventsByIds";
import type { ConceptDataset } from "./types";

export async function loadDatasets() {
    const [a, b] = await Promise.all([
        fetch("/umap/concept/LIBERTY.json").then((r) => r.json()),
        fetch("/umap/concept/PREROGATIVE.json").then((r) => r.json()),
    ]);

    // Augment each point with its concept name so colorBy:"concept" works.
    const augmented = (d: ConceptDataset) => ({
        ...d,
        points: d.points.map((p) => ({
            ...p,
            concept: d.concept
        }))
    });

    const datasetsTagged = [a, b].map(augmented);
    const allEventIds = [
        ...new Set(datasetsTagged.flatMap(d => d.points.map(p => p.event_id)))
    ];

    const events = await getEventsByIds(allEventIds);

    // Turn BigInt ids into strings
    const eventMap = new Map(
        events.map(e => [String(e.event_id), e])
    );

    // merge event fields into points
    const enrichedDatasets = datasetsTagged.map(d => ({
        ...d,
        points: d.points.map(p => {
            // Turn BigInt ids into strings
            const event = eventMap.get(String(p.event_id));
            return {
                ...p,
                ...(event ?? { whoops: true })
            };
        })
    }));

    return enrichedDatasets;
}