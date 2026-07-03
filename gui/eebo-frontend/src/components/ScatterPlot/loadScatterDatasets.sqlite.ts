import { execRows } from "../../services/db";
import type { YearMode } from "../../types";
import type { ConceptDatasetSqlite, PointData } from "./types";

type SqlFragment = {
    sql: string;
    params: any[];
};

function yearFilter(
    tablePrefix: string,
    yearMode: YearMode,
    fromYear: number,
    toYear: number
): SqlFragment {
    const col = `${ tablePrefix }pub_year`;

    if (yearMode === "single") {
        return {
            sql: `AND ${ col } = ?`,
            params: [fromYear],
        };
    }

    return {
        sql: `AND ${ col } BETWEEN ? AND ?`,
        params: [fromYear, toYear],
    };
}

function authorMatchFilter(str: string): SqlFragment {
    // escape LIKE wildcards only (still safe since we're parameterized)
    const term = str
        .replace(/\\/g, "\\\\")
        .replace(/%/g, "\\%")
        .replace(/_/g, "\\_");

    return {
        sql: `AND d.author LIKE ? ESCAPE '\\'`,
        params: [`%${ term }%`],
    };
}

interface LoadDatasetsParams {
    concepts: string[];
    fromYear: number;
    toYear: number;
    yearMode: YearMode;
    authorMatch: string;
    dataType: string;
}

export async function loadDatasets(
    params: LoadDatasetsParams
): Promise<ConceptDatasetSqlite[]> {
    console.log(
        `[loadDatasets] START ${ params.dataType } for`,
        params.concepts
    );

    const start = performance.now();

    try {
        const rv = await Promise.all(
            params.concepts.map(async (concept) => {
                let pointsQuery = "";
                let queryParams: any[] = [];

                if (params.dataType === "concept_neighbours") {
                    const year = yearFilter(
                        "n.",
                        params.yearMode,
                        params.fromYear,
                        params.toYear
                    );

                    const author =
                        params.authorMatch?.trim().length > 0
                            ? authorMatchFilter(params.authorMatch)
                            : null;

                    pointsQuery = `
                        SELECT
                            n.neighbour_event_id AS event_id,
                            n.token, n.doc_id, n.pub_year,
                            n.token_idx, n.window_id,
                            n.nx, n.ny, n.gnx, n.gny,
                            n.depth
                        FROM neighbours n
                        JOIN events e ON e.event_id = n.neighbour_event_id
                        JOIN documents d ON d.doc_id = n.doc_id
                        WHERE e.concept = ?
                          AND n.pub_year IS NOT NULL
                          ${ author ? author.sql : "" }
                          ${ year.sql }
                    `;

                    queryParams = [
                        concept,
                        ...(author ? author.params : []),
                        ...year.params,
                    ];
                } else if (params.dataType === "concept_clusters") {
                    pointsQuery = `
                        SELECT
                            cluster_id,
                            cluster_label,
                            centroid_nx as nx,
                            centroid_ny as ny,
                            centroid_gnx as gnx,
                            centroid_gny as gny,
                            point_count
                        FROM concept_cluster_info
                        WHERE concept = ?
                        ORDER BY cluster_id
                    `;

                    queryParams = [concept];
                } else {
                    const year = yearFilter(
                        "d.",
                        params.yearMode,
                        params.fromYear,
                        params.toYear
                    );

                    const author =
                        params.authorMatch?.trim().length > 0
                            ? authorMatchFilter(params.authorMatch)
                            : null;

                    pointsQuery = `
                        SELECT
                            event_id, token, d.doc_id, d.pub_year,
                            token_idx, window_id,
                            nx, ny, gnx, gny,
                            cluster_id, cluster_label
                        FROM events
                        JOIN documents d ON d.doc_id = events.doc_id
                        WHERE concept = ?
                          AND d.pub_year IS NOT NULL
                          ${ author ? author.sql : "" }
                          ${ year.sql }
                    `;

                    queryParams = [
                        concept,
                        ...(author ? author.params : []),
                        ...year.params,
                    ];
                }

                const points = await execRows(pointsQuery, queryParams);

                console.debug(
                    `[loadDatasets] ${ concept } | ${ params.dataType } | raw points: ${ points.length }`
                );

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
                        if (params.dataType === "concept_clusters") {
                            return {
                                event_id: `cluster-${ p[0] }`,
                                cluster_id: p[0],
                                cluster_label: p[1],
                                nx: p[2],
                                ny: p[3],
                                gnx: p[4],
                                gny: p[5],
                                point_count: p[6],
                                concept,

                                token: "_NULL_",
                                token_idx: -999,
                                doc_id: "_NULL_",
                                pub_year: -999,
                                lat: -999,
                                lng: -999,
                                vector_id: "_NULL_",
                                window_id: -999,
                                window_token_pos: -999,
                                windowKey: "_NULL_",
                            } as PointData;
                        } else {
                            return {
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
                                cluster_id: p[10],
                                cluster_label: p[11],
                                concept,
                            } as PointData;
                        }
                    }),
                } as ConceptDatasetSqlite;
            })
        );

        const duration = performance.now() - start;
        console.log(
            `[loadDatasets] FINISHED ${ params.dataType } in ${ duration.toFixed(1) }ms`
        );

        return rv;
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
    if (!params) return { type: "bfs_global", points: [] };

    console.time("[loadBfsDataset]");

    const year = yearFilter(
        "n.",
        params.yearMode,
        params.fromYear,
        params.toYear
    );

    const points = await execRows(
        `
        SELECT
            n.neighbour_event_id AS event_id,
            n.token, n.doc_id, n.pub_year,
            n.token_idx, n.window_id,
            n.nx, n.ny,
            n.gnx, n.gny,
            n.depth
        FROM neighbours n
        WHERE n.pub_year IS NOT NULL
          ${ year.sql }
        `,
        year.params
    );

    console.timeEnd("[loadBfsDataset]");

    return {
        type: "bfs_global",
        points: (points as any[]).map((p) => ({
            ...p,
            event_id: String(p.event_id),
        })),
    };
}