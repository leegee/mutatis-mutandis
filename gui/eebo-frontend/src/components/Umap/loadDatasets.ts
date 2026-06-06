import { getEventsByIds } from "../../services/db/getEventsByIds";
import { type YearMode } from "../../state/controls.store";
import type { ConceptDataset } from "./types";

interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
}

export async function loadDatasets(args: LoadDatasetsParams) {
    const conceptDatasetsRaw = await Promise.all(
        args.concepts.map((c: string) => {
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

    const matchesYear = (year: number) =>
        args.yearMode === "single"
            ? (y: number) => y === year
            : (y: number) => y >= args.fromYear && y <= args.toYear;

    const yearCheck = matchesYear(args.fromYear);

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