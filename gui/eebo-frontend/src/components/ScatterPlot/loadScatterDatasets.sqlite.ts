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

    console.debug(
        `[loadDatasets] START ${ params.dataType }`,
        params.concepts
    );

    const start = performance.now();

    try {
        const rv = await Promise.all(
            params.concepts.map(async (concept) => {
                let sql = "";
                let queryParams: any[] = [];

                if (params.dataType === "concept_neighbours") {
                    const year = yearFilter(
                        "neighbourhood_event.",
                        params.yearMode,
                        params.fromYear,
                        params.toYear
                    );

                    const author = params.authorMatch?.trim()
                        ? authorMatchFilter(params.authorMatch)
                        : null;

                    sql = `SELECT
                            neighbourhood_event.event_id,
                            neighbourhood_event.token,
                            neighbourhood_event.doc_id,
                            neighbourhood_event.pub_year,
                            neighbourhood_event.token_idx,
                            neighbourhood_event.window_id,
                            neighbourhood_event.nx,
                            neighbourhood_event.ny,
                            neighbourhood_event.gnx,
                            neighbourhood_event.gny,
                            neighbourhood_event.cluster_id,
                            neighbourhood_event.cluster_label,
                            n.depth,
                            neighbourhood_event.corpus
                        FROM concept_field_events seed
                            JOIN neighbours n
                                ON n.event_id = seed.event_id
                            JOIN events neighbourhood_event
                                ON neighbourhood_event.event_id = n.neighbour_event_id
                            LEFT JOIN documents d
                                ON d.doc_id = neighbourhood_event.doc_id
                                AND d.corpus = neighbourhood_event.corpus
                            WHERE seed.concept = ?
                            AND seed.role = 'seed'
                          ${ author ? author.sql : "" }
                          ${ year.sql }
                    `;

                    queryParams = [
                        concept,
                        ...(author ? author.params : []),
                        ...year.params,
                    ];

                }

                else if (params.dataType === "concept_clusters") {
                    const year = yearFilter(
                        "e.",
                        params.yearMode,
                        params.fromYear,
                        params.toYear
                    );

                    sql = `
                        SELECT DISTINCT
                            e.corpus,
                            c.cluster_id,
                            c.cluster_label,
                            c.centroid_nx AS nx,
                            c.centroid_ny AS ny,
                            c.centroid_gnx AS gnx,
                            c.centroid_gny AS gny,
                            c.point_count,
                            c.description
                        FROM concept_cluster_info c
                            JOIN concept_field_events f
                                ON f.concept = c.concept
                            JOIN events e
                                ON e.event_id = f.event_id
                        WHERE c.concept = ?
                        AND f.role = 'seed'
                        AND e.cluster_id = c.cluster_id
                          ${ year.sql }
                        ORDER BY c.cluster_id
                    `;

                    queryParams = [concept, ...year.params];
                }

                else {
                    const year = yearFilter(
                        "e.",
                        params.yearMode,
                        params.fromYear,
                        params.toYear
                    );

                    const author =
                        params.authorMatch?.trim()
                            ? authorMatchFilter(params.authorMatch)
                            : null;

                    sql = `
                        SELECT
                            e.event_id,
                            e.token,
                            d.doc_id,
                            d.pub_year,
                            e.token_idx,
                            e.window_id,
                            e.nx,
                            e.ny,
                            e.gnx,
                            e.gny,
                            e.cluster_id,
                            e.cluster_label,
                            e.corpus
                        FROM concept_field_events f
                            JOIN events e
                                ON e.event_id = f.event_id
                            LEFT JOIN documents d
                                ON d.doc_id = e.doc_id
                                AND d.corpus = e.corpus
                            WHERE f.concept = ?
                            AND f.role = 'seed'
                          ${ author ? author.sql : "" }
                          ${ year.sql }
                    `;

                    queryParams = [
                        concept,
                        ...(author ? author.params : []),
                        ...year.params,
                    ];
                }

                const points = await execRows(sql, queryParams);

                console.debug(`[loadDatasets] ${ concept } | ${ params.dataType } | ${ points.length }`);

                return {
                    concept,
                    type: params.dataType,
                    points: (points as any[]).map((p: any[]) => {
                        if (params.dataType === "concept_clusters") {
                            return {
                                event_id: `${ concept }-cluster-${ p[1] }-${ p[0] }`,
                                cluster_id: p[1],
                                cluster_label: p[2],
                                nx: p[3],
                                ny: p[4],
                                gnx: p[5],
                                gny: p[6],
                                point_count: p[7],
                                label: p[2],
                                description: p[8],
                                concept,
                                corpus: p[0],
                                token: "_NULL_",
                                token_idx: -999,
                                doc_id: "_NULL_",
                                pub_year: -999,
                                vector_id: "_NULL_",
                                window_id: -999,
                                window_token_pos: -999,
                                windowKey: "_NULL_",
                            } as PointData;
                        }

                        // concept_neighbours has an extra `n.depth` column before
                        // `corpus`, so the corpus index shifts for that branch.
                        const corpusIdx = params.dataType === "concept_neighbours" ? 13 : 12;

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
                            depth: p[12],
                            corpus: p[corpusIdx],
                            concept,
                        } as PointData;
                    }),
                } as ConceptDatasetSqlite;

            })
        );

        console.debug(`[loadDatasets] FINISHED in ${ (performance.now() - start).toFixed(1) }ms`);
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

    if (!params) {
        return {
            type: "bfs_global",
            points: [],
        };
    }

    console.time("[loadBfsDataset]");

    const year = yearFilter(
        "neighbourhood_event.",
        params.yearMode,
        params.fromYear,
        params.toYear
    );

    const points = await execRows(`
        SELECT
            neighbourhood_event.event_id,
            neighbourhood_event.token,
            neighbourhood_event.doc_id,
            neighbourhood_event.pub_year,
            neighbourhood_event.token_idx,
            neighbourhood_event.window_id,
            neighbourhood_event.nx,
            neighbourhood_event.ny,
            neighbourhood_event.gnx,
            neighbourhood_event.gny,
            n.depth,
            neighbourhood_event.corpus
        FROM neighbours n
        JOIN events neighbourhood_event
            ON neighbourhood_event.event_id = n.neighbour_event_id
        WHERE neighbourhood_event.pub_year IS NOT NULL
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
