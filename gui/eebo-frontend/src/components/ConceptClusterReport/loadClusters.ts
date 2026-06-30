import { execRows } from "../../services/db";
import type { YearMode } from "../../types";
import type { ClusterInfo } from "./ConceptClusters";

export interface ClusterAggregate {
    top_tokens: [string, number][];
    top_docs: [string, number][];
}

export interface ClusterPoint {
    event_id: string;
    cluster_id: number;
    cluster_label: string | null;
    nx: number;
    ny: number;
    gnx: number;
    gny: number;
}

export interface ClusterFile {
    type: string;
    concept: string;
    generated_at: string;
    n_events: number;
    bounds: Record<string, number>;
    globalBounds: Record<string, number>;
    clusters: {
        label_map: Record<string, string>;
        aggregates: Record<string, ClusterAggregate>;
    };
    points: ClusterInfo[];
}

export interface LoadClusterFileParams {
    concept: string;
    // Omit (or pass yearMode: undefined) for the unfiltered, full-corpus
    // view that reads pre-computed concept_aggregate rows.
    yearMode?: YearMode;
    fromYear?: number;
    toYear?: number;
}

const TOP_N = 25; // matches top_n used by tier3's compute_cluster_aggregates

function yearFilter(yearMode: YearMode, fromYear: number, toYear: number): string {
    return yearMode === "single"
        ? `AND pub_year = ${ fromYear }`
        : `AND pub_year BETWEEN ${ fromYear } AND ${ toYear }`;
}

// Bucket [cluster_id, key, count] rows into { [cluster_id]: [key, count][] },
// sorted desc by count and capped to TOP_N per cluster. Used only for the
// live-aggregation (year-filtered) path; the unfiltered path reads
// pre-ranked rows straight from concept_aggregate instead.
function bucketTopN(rows: any[][]): Record<string, [string, number][]> {
    const byCluster = new Map<string, [string, number][]>();
    for (const [cid, key, count] of rows) {
        const cidStr = String(cid);
        const arr = byCluster.get(cidStr) ?? [];
        arr.push([key, count]);
        byCluster.set(cidStr, arr);
    }
    const out: Record<string, [string, number][]> = {};
    for (const [cid, arr] of byCluster) {
        arr.sort((a, b) => b[1] - a[1]);
        out[cid] = arr.slice(0, TOP_N);
    }
    return out;
}


export async function loadClusters(params: LoadClusterFileParams): Promise<ClusterFile> {
    const { concept, yearMode, fromYear, toYear } = params;
    const yearActive = yearMode != null && fromYear != null && toYear != null;
    const yearClause = yearActive ? yearFilter(yearMode!, fromYear!, toYear!) : "";

    console.log(`[loadClusterFile] START for ${ concept } (yearFiltered=${ yearActive })`);
    const start = performance.now();

    // 1. Per-event points belonging to a cluster, optionally year-filtered
    const pointRows = await execRows(
        `SELECT event_id, cluster_id, cluster_label, nx, ny, gnx, gny
         FROM events
         WHERE concept = ?
           AND cluster_id IS NOT NULL
           AND pub_year IS NOT NULL
           ${ yearClause }`,
        [concept]
    );

    const points: ClusterPoint[] = (pointRows as any[]).map((p: any[]) => ({
        event_id: String(p[0]),
        cluster_id: p[1],
        cluster_label: p[2],
        nx: p[3],
        ny: p[4],
        gnx: p[5],
        gny: p[6],
    }));

    // 2. label_map: cluster_id -> cluster_label, derived from the points
    const label_map: Record<string, string> = {};
    for (const p of points) {
        const cidStr = String(p.cluster_id);
        if (p.cluster_label && !(cidStr in label_map)) {
            label_map[cidStr] = p.cluster_label;
        }
    }

    // 3. top_tokens / top_docs per cluster
    let aggregates: Record<string, ClusterAggregate>;

    if (!yearActive) {
        // Fast path: pre-computed, pre-ranked rows written by tier3's
        // compute_cluster_aggregates (rank starts at 1 there, but we sort
        // by rank regardless of starting value so that's not load-bearing).
        const aggRows = await execRows(
            `SELECT cluster_id, kind, value, count
             FROM concept_aggregate
             WHERE concept = ?
               AND cluster_id IS NOT NULL
               AND kind IN ('token', 'doc')
             ORDER BY cluster_id, kind, rank`,
            [concept]
        );

        aggregates = {};
        for (const [cid, kind, value, count] of aggRows as any[]) {
            const cidStr = String(cid);
            if (!aggregates[cidStr]) {
                aggregates[cidStr] = { top_tokens: [], top_docs: [] };
            }
            if (kind === "token") {
                aggregates[cidStr].top_tokens.push([value, count]);
            } else if (kind === "doc") {
                aggregates[cidStr].top_docs.push([value, count]);
            }
        }
    } else {
        // Year-filtered path: concept_aggregate has no pub_year column
        // (counts are baked across the whole corpus at cluster-build
        // time), so live-aggregate over `events` instead, scoped to the
        // same year range as the points query above.
        const tokenRows = await execRows(
            `SELECT cluster_id, token, COUNT(*) as c
             FROM events
             WHERE concept = ?
               AND cluster_id IS NOT NULL
               AND pub_year IS NOT NULL
               ${ yearClause }
             GROUP BY cluster_id, token`,
            [concept]
        );

        const docRows = await execRows(
            `SELECT cluster_id, doc_id, COUNT(*) as c
             FROM events
             WHERE concept = ?
               AND cluster_id IS NOT NULL
               AND pub_year IS NOT NULL
               ${ yearClause }
             GROUP BY cluster_id, doc_id`,
            [concept]
        );

        const topTokensByCluster = bucketTopN(tokenRows as any[]);
        const topDocsByCluster = bucketTopN(docRows as any[]);

        aggregates = {};
        const allClusterIds = new Set([
            ...Object.keys(topTokensByCluster),
            ...Object.keys(topDocsByCluster),
        ]);
        for (const cid of allClusterIds) {
            aggregates[cid] = {
                top_tokens: topTokensByCluster[cid] ?? [],
                top_docs: topDocsByCluster[cid] ?? [],
            };
        }
    }

    // Make sure every cluster present in `points` has an aggregates entry,
    for (const cidStr of Object.keys(label_map)) {
        if (!aggregates[cidStr]) {
            aggregates[cidStr] = { top_tokens: [], top_docs: [] };
        }
    }

    // 4. Bounds (not year-scoped — local/global projection bounds are
    // fixed at tier3 build time for the whole concept, independent of any
    // year filter applied at read time)
    const boundsRows = await execRows(
        `SELECT local_min_x, local_max_x, local_min_y, local_max_y,
                global_min_x, global_max_x, global_min_y, global_max_y
         FROM concept_projection_bounds WHERE concept = ?`,
        [concept]
    );
    const b = (boundsRows as any[])[0] || [];

    const duration = performance.now() - start;
    console.log(`[loadClusterFile] FINISHED in ${ duration.toFixed(1) }ms`);

    return {
        type: "concept_clusters",
        concept,
        generated_at: new Date().toISOString(),
        n_events: points.length,
        bounds: {
            minX: b[0] ?? 0, maxX: b[1] ?? 0,
            minY: b[2] ?? 0, maxY: b[3] ?? 0,
        },
        globalBounds: {
            minX: b[4] ?? 0, maxX: b[5] ?? 0,
            minY: b[6] ?? 0, maxY: b[7] ?? 0,
        },
        clusters: { label_map, aggregates },
        points,
    };
}

