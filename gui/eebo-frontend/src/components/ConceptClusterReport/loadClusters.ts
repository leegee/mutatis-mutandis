import { execRows } from "../../services/db";

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
    points: ClusterPoint[];
}

export async function loadClusterFile(concept: string): Promise<ClusterFile> {
    console.log(`[loadClusterFile] START for ${ concept }`);
    const start = performance.now();

    // 1. Per-event points belonging to a cluster
    const pointRows = await execRows(
        `SELECT event_id, cluster_id, cluster_label, nx, ny, gnx, gny
         FROM events
         WHERE concept = ?
           AND cluster_id IS NOT NULL
           AND pub_year IS NOT NULL`,
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

    // 3. Per-cluster top_tokens / top_docs, pre-ranked by the writer
    //    (tier2_0_concept_events.py's _aggregate_rows). Rows are already
    //    ordered by rank ascending per (concept, cluster_id, kind), so we
    //    just need to bucket them in the order they arrive.
    const aggRows = await execRows(
        `SELECT cluster_id, kind, value, count
         FROM concept_aggregate
         WHERE concept = ?
           AND cluster_id IS NOT NULL
           AND kind IN ('token', 'doc')
         ORDER BY cluster_id, kind, rank`,
        [concept]
    );

    const aggregates: Record<string, ClusterAggregate> = {};
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

    // Make sure every cluster present in `points` has an aggregates entry,
    // even if concept_aggregate has no rows for it yet (e.g. cluster_id
    // wasn't populated when the aggregate rows were written).
    for (const cidStr of Object.keys(label_map)) {
        if (!aggregates[cidStr]) {
            aggregates[cidStr] = { top_tokens: [], top_docs: [] };
        }
    }

    // 4. Bounds
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
