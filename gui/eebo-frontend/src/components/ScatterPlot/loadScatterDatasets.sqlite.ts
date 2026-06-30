import { execRows } from "../../services/db";
import { controls } from "../../state/controls.store";
import type { YearMode } from "../../types";
import type { ConceptDatasetJSON, ConceptDatasetSqlite, PointData } from "./types";

function yearFilter(tablePrefix: string, yearMode: YearMode, fromYear: number, toYear: number): string {
    const col = `${ tablePrefix }pub_year`;
    return yearMode === "single"
        ? `AND ${ col } = ${ fromYear }`
        : `AND ${ col } BETWEEN ${ fromYear } AND ${ toYear }`;
}

interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
    dataType: string;
}


export async function loadDatasets(params: LoadDatasetsParams): Promise<ConceptDatasetSqlite[]> {
    console.log(`[loadDatasets] START ${ params.dataType } for`, params.concepts);
    const start = performance.now();

    try {
        const rv = await Promise.all(
            params.concepts.map(async (concept) => {
                let pointsQuery;
                if (params.dataType === "concept_neighbours") {
                    pointsQuery = `SELECT
                            n.neighbour_event_id AS event_id,
                            n.token, n.doc_id, n.pub_year,
                            n.token_idx, n.window_id,
                            n.nx, n.ny, n.gnx, n.gny,
                            n.depth
                       FROM neighbours n
                       JOIN events e ON e.event_id = n.neighbour_event_id
                       WHERE e.concept = ?
                         AND n.pub_year IS NOT NULL
                         ${ yearFilter('n.', params.yearMode, params.fromYear, params.toYear) }
                    `;
                }
                else if (params.dataType === "concept_clusters") {
                    pointsQuery = `SELECT
                        event_id, token, doc_id, pub_year,
                        token_idx, window_id,
                        nx, ny, gnx, gny,
                        cluster_id, cluster_label
                        FROM events
                        WHERE concept = ?
                        AND pub_year IS NOT NULL
                        ${ yearFilter('', params.yearMode, params.fromYear, params.toYear) }
                    `;
                }
                else {
                    pointsQuery = `SELECT
                            event_id, token, doc_id, pub_year,
                            token_idx, window_id,
                            nx, ny, gnx, gny,
                            cluster_id, cluster_label
                       FROM events
                       WHERE concept = ?
                         AND pub_year IS NOT NULL
                         ${ yearFilter('', params.yearMode, params.fromYear, params.toYear) }
                    `;
                }

                const points = await execRows(pointsQuery, [concept]);

                console.debug(`[loadDatasets] ${ concept } | ${ params.dataType } | raw points: ${ points.length }`);
                if (points.length > 0) {
                    console.debug("Sample point:", points[0]);
                }

                const boundsRows = await execRows(
                    `SELECT local_min_x, local_max_x, local_min_y, local_max_y,
                            global_min_x, global_max_x, global_min_y, global_max_y
                     FROM concept_projection_bounds WHERE concept = ?`,
                    [concept]
                );

                const b = (boundsRows as any[])[0] || {};

                return {
                    concept,
                    type: params.dataType,
                    bounds: {
                        minX: b[0] ?? 0,
                        maxX: b[1] ?? 0,
                        minY: b[2] ?? 0,
                        maxY: b[3] ?? 0,
                    },
                    globalBounds: {
                        minX: b[4] ?? 0,
                        maxX: b[5] ?? 0,
                        minY: b[6] ?? 0,
                        maxY: b[7] ?? 0,
                    },
                    points: (points as any[]).map((p: any[]) => {
                        const base = {
                            event_id: String(p[0]),
                            token: p[1],
                            doc_id: p[2],
                            pub_year: p[3],
                            token_idx: p[4],
                            window_id: p[5],
                            nx: p[6],
                            ny: p[7],
                            gnx: p[8],
                            gny: p[9],
                            concept,
                        };

                        if (params.dataType === "concept_neighbours") {
                            return {
                                ...base,
                                depth: p[10],
                            } as PointData;
                        } else {
                            return {
                                ...base,
                                cluster_id: p[10],
                                cluster_label: p[11],
                            } as PointData;
                        }
                    }),
                } as ConceptDatasetSqlite;
            })
        );

        const duration = performance.now() - start;
        console.log(`[loadDatasets] Total load time for ${ params.concepts.join(', ') }: ${ duration.toFixed(1) }ms`);
        return rv;
    }

    catch (error) {
        console.error("loadDatasets error", error);
        return [];
    }
}


export async function loadBfsDataset(params: {
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
}) {
    if (!params) return { type: "bfs_global", points: [] };

    console.time('[loadBfsDataset]')
    const yearWhere = yearFilter('n.', params.yearMode, params.fromYear, params.toYear);

    const points = await execRows(
        `SELECT
            n.neighbour_event_id AS event_id,
            n.token, n.doc_id, n.pub_year,
            n.token_idx, n.window_id,
            n.nx, n.ny,
            n.gnx, n.gny,
            n.depth
         FROM neighbours n
         WHERE n.pub_year IS NOT NULL
           ${ yearWhere }`,
        []
    );

    console.timeEnd('[loadBfsDataset]');

    return {
        type: "bfs_global",
        points: (points as any[]).map(p => ({
            ...p,
            event_id: String(p.event_id),
        })),
    };
}
