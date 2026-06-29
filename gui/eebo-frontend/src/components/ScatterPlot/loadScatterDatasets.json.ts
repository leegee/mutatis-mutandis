import { loadJson } from "../../lib/json";
import { getEventsByIds } from "../../services/db";
import type { YearMode } from "../../types";
import type { ConceptDatasetJSON } from "./types";

function matchesYearFactory(params: {
    yearMode: YearMode;
    fromYear: number;
    toYear: number;
}) {
    return params.yearMode === "single"
        ? (y: number) => y === params.fromYear
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
        // event (DB row) spreads first so JSON point fields win all
        // collisions — in particular `depth` from the JSON is never
        // clobbered by a same-named DB column.
        acc.push({ ...event, ...p });
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

export async function loadDatasets(params: LoadDatasetsParams): Promise<ConceptDatasetJSON[]> {
    try {
        // All three data types share the same file shape — only the path differs.
        const rawDatasets = await Promise.all(
            params.concepts.map((c: string) =>
                loadJson<ConceptDatasetJSON>(
                    `/data/scatter/${ params.dataType }/${ c }.json`,
                    c
                )
            )
        );

        // Tag each point with its concept (already present in cluster files,
        // but harmless to overwrite with the canonical value).
        const tagged = rawDatasets.map((d: ConceptDatasetJSON) => ({
            ...d,
            points: d.points.map(p => ({ ...p, concept: d.concept })),
        }));

        const allEventIds = tagged.flatMap(d => d.points.map(p => String(p.event_id)));
        const eventMap = await buildEventMap(allEventIds);
        const yearCheck = matchesYearFactory(params);
        const rv = tagged.map(d => ({
            ...d,
            points: enrichPoints(d.points, eventMap, yearCheck),
        }));
        return rv;
    }
    catch (error) {
        console.error('xxxxxxxxxxxxxx', error);
    }
    return [];
}


interface LoadBfsParams {
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
}

export async function loadBfsDataset(params: LoadBfsParams) {
    const bfs = await loadJson(
        "/data/scatter/bfs_global/global.json",
        "bfs_global"
    );

    const eventIds = bfs.points.map((p: { event_id: string }) => String(p.event_id));
    const eventMap = await buildEventMap(eventIds);
    const yearCheck = matchesYearFactory(params);

    return {
        ...bfs,
        points: enrichPoints(bfs.points, eventMap, yearCheck),
    };
}
