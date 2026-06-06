import { getEventsByIds } from "../../services/db/getEventsByIds";
import { type YearMode } from "../../state/controls.store";
import type { ConceptDataset } from "./types";

interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
    dataType: string;
}

export async function loadDatasets(params: LoadDatasetsParams) {
    const conceptDatasetsRaw = await Promise.all(
        params.concepts.map((c: string) => {
            return fetch(`/umap/${ params.dataType }/${ c }.json`).then((r) => r.json());
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

    const matchesYear = (year: number) =>
        params.yearMode === "single"
            ? (y: number) => y === year
            : (y: number) => y >= params.fromYear && y <= params.toYear;

    const yearCheck = matchesYear(params.fromYear);

    const enrichedDatasets = datasetsTagged.map(d => ({
        ...d,
        points: d.points.reduce<typeof d.points>((acc, p) => {
            const event = eventMap.get(String(p.event_id));
            const pubYear = event?.pub_year;

            if (pubYear == null || !yearCheck(pubYear)) return acc;

            acc.push({
                ...p,
                ...event,
            });

            return acc;
        }, [])
    }));

    return enrichedDatasets;
}