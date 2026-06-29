import { execRows } from "../../services/db";
import type { YearMode } from "../../types";
import type { ConceptDatasetJSON, ConceptDatasetSqlite } from "./types";

function yearFilter(yearMode: YearMode, fromYear: number, toYear: number): string {
    return yearMode === "single"
        ? `AND pub_year = ${ fromYear }`
        : `AND pub_year BETWEEN ${ fromYear } AND ${ toYear }`;
}

interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
    dataType: string;
}

export async function loadDatasets(params: LoadDatasetsParams): Promise<ConceptDatasetSqlite[]> {
    try {
        const yearWhere = yearFilter(params.yearMode, params.fromYear, params.toYear);

        return await Promise.all(
            params.concepts.map(async (concept) => {
                const isNeighbours = params.dataType === "concept_neighbours";

                const points = isNeighbours
                    ? await execRows(
                        `SELECT
                            n.neighbour_event_id AS event_id,
                            n.token, n.doc_id, n.pub_year,
                            n.token_idx, n.window_id,
                            n.local_x, n.local_y,
                            n.global_x, n.global_y,
                            n.depth
                         FROM neighbours n
                         JOIN events e ON e.event_id = n.event_id
                         WHERE e.concept = ?
                           AND n.pub_year IS NOT NULL
                           ${ yearWhere }`,
                        [concept]
                    )
                    : await execRows(
                        `SELECT
                            event_id, token, doc_id, pub_year,
                            token_idx, window_id,
                            local_x, local_y,
                            global_x, global_y,
                            cluster_id, cluster_label
                         FROM events
                         WHERE concept = ?
                           AND pub_year IS NOT NULL
                           ${ yearWhere }`,
                        [concept]
                    );

                const bounds = await execRows(
                    `SELECT
                        local_min_x, local_max_x, local_min_y, local_max_y,
                        global_min_x, global_max_x, global_min_y, global_max_y
                     FROM concept_projection_bounds
                     WHERE concept = ?`,
                    [concept]
                );

                const boundsRow = (bounds as any[])[0] ?? null;

                return {
                    concept,
                    type: params.dataType,
                    bounds: {
                        minX: boundsRow.local_min_x,
                        maxX: boundsRow.local_max_x,
                        minY: boundsRow.local_min_y,
                        maxY: boundsRow.local_max_y,
                    },
                    globalBounds: {
                        minX: boundsRow.global_min_x,
                        maxX: boundsRow.global_max_x,
                        minY: boundsRow.global_min_y,
                        maxY: boundsRow.global_max_y,
                    },
                    points: (points as any[]).map(p => ({
                        ...p,
                        event_id: String(p.event_id),
                        concept,
                    })),
                };
            })
        );
    } catch (error) {
        console.error("loadDatasets error", error);
        return [];
    }
}

export async function loadBfsDataset(params: {
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
}) {
    const yearWhere = yearFilter(params.yearMode, params.fromYear, params.toYear);

    const points = await execRows(
        `SELECT
            n.neighbour_event_id AS event_id,
            n.token, n.doc_id, n.pub_year,
            n.token_idx, n.window_id,
            n.local_x, n.local_y,
            n.global_x, n.global_y,
            n.depth
         FROM neighbours n
         WHERE n.pub_year IS NOT NULL
           ${ yearWhere }`,
        []
    );

    return {
        type: "bfs_global",
        points: (points as any[]).map(p => ({
            ...p,
            event_id: String(p.event_id),
        })),
    };
}
