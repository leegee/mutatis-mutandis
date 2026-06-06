import { getEventsByIds } from "../../services/db/getEventsByIds";
import type { ConceptDataset } from "./types";

export async function loadDatasets(concepts: string[]) {
    const conceptDatasetsRaw = await Promise.all(
        concepts.map((c: string) => {
            return fetch(`/umap/concept/${ c }.json`).then((r) => r.json());
        })
    );

    // Augment each point with its concept name so colorBy:"concept" works.
    const augmented = (d: ConceptDataset) => ({
        ...d,
        points: d.points.map((p) => ({
            ...p,
            concept: d.concept
        }))
    });

    const datasetsTagged = conceptDatasetsRaw.map(augmented);
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
            console.log(event?.pub_year)
            return {
                ...p,
                ...(event ?? { whoops: true })
            };
        })
    }));

    return enrichedDatasets;
}