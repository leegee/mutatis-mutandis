import { getEventsByIds } from "../../services/db/getEventsByIds";
import { type YearMode } from "../../state/controls.store";
import type { ConceptDataset } from "./types";


function matchesYearFactory(params: {
    yearMode: YearMode;
    fromYear: number;
    toYear: number;
}) {
    return (year: number) =>
        params.yearMode === "single"
            ? (y: number) => y === year
            : (y: number) => y >= params.fromYear && y <= params.toYear;
}


async function buildEventMap(eventIds: string[]) {
    const events = await getEventsByIds([...new Set(eventIds)]);
    return new Map(events.map(e => [String(e.event_id), e]));
}


function enrichPoints<T extends { event_id: string }>(
    points: T[],
    eventMap: Map<string, any>,
    yearCheck: (y: number) => boolean
) {
    return points.reduce<any[]>((acc, p) => {
        const event = eventMap.get(String(p.event_id));
        const pubYear = event?.pub_year;

        if (pubYear == null || !yearCheck(pubYear)) return acc;

        acc.push({
            ...p,
            ...event,
        });

        return acc;
    }, []);
}


interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
    dataType: string;
}

export async function loadDatasets(params: LoadDatasetsParams) {

    if (params.dataType === 'concept_clusters') {
        return await loadConceptClusters(params.concepts);
    }

    const conceptDatasetsRaw = await Promise.all(
        params.concepts.map((c: string) =>
            fetch(`/umap/${ params.dataType }/${ c }.json`).then(r => r.json())
        )
    );

    const augmented = (d: ConceptDataset) => ({
        ...d,
        points: d.points.map(p => ({
            ...p,
            concept: d.concept,
        })),
    });

    const datasetsTagged = conceptDatasetsRaw.map(augmented);
    const allEventIds = datasetsTagged.flatMap(d =>
        d.points.map(p => p.event_id)
    );
    const eventMap = await buildEventMap(allEventIds);
    const yearCheck = matchesYearFactory(params)(params.fromYear);
    const enrichedDatasets = datasetsTagged.map(d => ({
        ...d,
        points: enrichPoints(d.points, eventMap, yearCheck),
    }));

    return enrichedDatasets;
}


interface LoadBfsParams {
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
}

interface BfsDataset {
    type: "bfs_global";
    bounds: any;
    globalBounds: any;
    depth: number;
    k: number;
    points: {
        event_id: string;
        x: number;
        y: number;
        nx: number;
        ny: number;
        gx: number;
        gy: number;
        gnx: number;
        gny: number;
    }[];
}

export async function loadBfsDataset(params: LoadBfsParams) {
    const bfs: BfsDataset = await fetch(
        `/umap/bfs_global/global.json`
    ).then(r => r.json());

    const eventIds = bfs.points.map(p => p.event_id);
    const eventMap = await buildEventMap(eventIds);

    const yearCheck = matchesYearFactory(params)(params.fromYear);

    return {
        ...bfs,
        points: enrichPoints(bfs.points, eventMap, yearCheck),
    };
}

// Add this helper
export async function loadConceptClusters(concepts: string[] | string): Promise<any[]> {
    const conceptList = Array.isArray(concepts) ? concepts : [concepts];
    const results: any[] = [];

    for (const concept of conceptList) {
        try {
            const filename = `${ concept.toLowerCase() }.json`;
            const response = await fetch(`/data/clusters/${ filename }`); // adjust path if needed

            if (!response.ok) {
                console.warn(`No cluster file for ${ concept }`);
                continue;
            }

            const data = await response.json();
            results.push(data);
        } catch (err) {
            console.warn(`Failed to load clusters for ${ concept }:`, err);
        }
    }

    return results;
}
