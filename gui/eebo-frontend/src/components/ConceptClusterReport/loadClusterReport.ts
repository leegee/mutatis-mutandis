import { execRows } from "../../services/db";
import type { YearMode } from "../../types";

export interface DocMeta {
    author: string | null;
    title: string | null;
    pub_year: number | null;
}

export interface ClusterSummary {
    id: number;
    label: string | null;
    // description: string | null;
    eventCount: number;
    topTokens: [string, number][];
    topDocs: [string, number][];
}

export interface ClusterReport {
    concept: string;
    generated_at: string;
    clusters: ClusterSummary[];
    docMeta: Record<string, DocMeta>;

    docExemplars: Record<
        number,
        {
            event_id: string;
            doc_id: string;
            token_idx: number;
        }[]
    >;
}

type SqlFragment = {
    sql: string;
    params: any[];
};

function yearFilter(
    yearMode: YearMode,
    fromYear: number,
    toYear: number
): SqlFragment {
    if (yearMode === "single") {
        return {
            sql: `AND e.pub_year = ?`,
            params: [fromYear],
        };
    }

    return {
        sql: `AND e.pub_year BETWEEN ? AND ?`,
        params: [fromYear, toYear],
    };
}

function authorFilter(str: string): SqlFragment {
    const term = str
        .replace(/\\/g, "\\\\")
        .replace(/%/g, "\\%")
        .replace(/_/g, "\\_");

    return {
        sql: `AND d.author LIKE ? ESCAPE '\\'`,
        params: [`%${ term }%`],
    };
}

interface loadClusterReportParams {
    concept: string;
    yearMode?: YearMode;
    fromYear?: number;
    toYear?: number;
    authorMatch?: string;
}



export async function loadClusterReport(
    params: loadClusterReportParams
): Promise<ClusterReport & { diagnostics: any }> {
    const { concept, yearMode, fromYear, toYear, authorMatch } = params;

    const yearActive = yearMode != null && fromYear != null && toYear != null;

    const year = yearActive
        ? yearFilter(yearMode!, fromYear!, toYear!)
        : null;

    const author = authorMatch?.trim()
        ? authorFilter(authorMatch)
        : null;

    const joins = `
    JOIN concept_field_events f
        ON f.event_id = e.event_id
    ${ author ? "JOIN documents d ON d.doc_id = e.doc_id" : "" }
    `;

    const authorSql = author ? author.sql : "";

    const baseWhere = `
        WHERE f.concept = ?
        AND f.role = 'seed'
        AND e.cluster_id IS NOT NULL
        AND e.pub_year IS NOT NULL
        ${ year ? year.sql : "" }
        ${ authorSql }
    `;

    const baseParams = [
        concept,
        ...(year ? year.params : []),
        ...(author ? author.params : []),
    ];

    // CLUSTER SUMMARY
    const clusterRows = await execRows(`
        SELECT
            e.cluster_id,
            c.cluster_label,
            COUNT(*) AS n
        FROM concept_field_events f
        JOIN events e
            ON e.event_id = f.event_id
        JOIN concept_cluster_info c
            ON c.concept = f.concept
            AND c.cluster_id = e.cluster_id
        ${ author ? "JOIN documents d ON d.doc_id = e.doc_id" : "" }
        WHERE f.concept = ?
        AND f.role = 'seed'
        AND e.pub_year IS NOT NULL
        AND e.cluster_id IS NOT NULL
        ${ year ? year.sql : "" }
        ${ authorSql }
        GROUP BY e.cluster_id, c.cluster_label
        ORDER BY e.cluster_id
        `,
        baseParams
    );
    const clusters = new Map<number, ClusterSummary & { eventIds: string[] }>();

    for (const [cid, label, n] of clusterRows as any[]) {
        clusters.set(cid, {
            id: cid,
            label,
            eventCount: n,
            topTokens: [],
            topDocs: [],
            eventIds: [],
        });
    }

    if (clusters.size === 0) {
        return {
            concept,
            generated_at: new Date().toISOString(),
            clusters: [],
            docMeta: {},
            docExemplars: {},
            diagnostics: {
                eventClusters: {},
                docClusterFootprint: {},
                clusterStats: {
                    totalEvents: 0,
                    multiClusterEvents: 0,
                    multiClusterRate: 0,
                },
                multiClusterEvents: [],
            },
        };
    }

    // -------------------------------------------------------
    // CLUSTER MEMBERSHIP (EVENT LEVEL)
    // -------------------------------------------------------
    const eventRows = await execRows(
        `
        SELECT e.event_id, e.cluster_id, e.doc_id
        FROM events e
        ${ joins }
        ${ baseWhere }
        `,
        baseParams
    );

    const eventClusters: Record<string, number[]> = {};
    const eventToDoc: Record<string, string> = {};

    for (const [event_id, cluster_id, doc_id] of eventRows as any[]) {
        const sid = String(event_id);

        if (!eventClusters[sid]) eventClusters[sid] = [];
        eventClusters[sid].push(cluster_id);

        eventToDoc[sid] = doc_id;
    }

    // populate cluster event lists
    for (const [eid, clustersArr] of Object.entries(eventClusters)) {
        const lastCluster = clustersArr[0];
        clusters.get(lastCluster)?.eventIds.push(eid);
    }

    // -------------------------------------------------------
    // TOP TOKENS
    // -------------------------------------------------------
    const tokenRows = await execRows(
        `
        SELECT cluster_id, token, c
        FROM (
            SELECT
                e.cluster_id,
                e.token,
                COUNT(*) AS c,
                ROW_NUMBER() OVER (
                    PARTITION BY e.cluster_id
                    ORDER BY COUNT(*) DESC
                ) AS rn
            FROM events e
            ${ joins }
            ${ baseWhere }
            GROUP BY e.cluster_id, e.token
        )
        WHERE rn <= 25
        `,
        baseParams
    );

    for (const [cid, token, c] of tokenRows as any[]) {
        clusters.get(cid)?.topTokens.push([token, c]);
    }

    // -------------------------------------------------------
    // TOP DOCS
    // -------------------------------------------------------
    const docRows = await execRows(
        `
        SELECT cluster_id, doc_id, c
        FROM (
            SELECT
                e.cluster_id,
                e.doc_id,
                COUNT(*) AS c,
                ROW_NUMBER() OVER (
                    PARTITION BY e.cluster_id
                    ORDER BY COUNT(*) DESC
                ) AS rn
            FROM events e
            ${ joins }
            ${ baseWhere }
            GROUP BY e.cluster_id, e.doc_id
        )
        WHERE rn <= 25
        `,
        baseParams
    );

    const docIds = new Set<string>();

    for (const [cid, doc_id, count] of docRows as any[]) {
        clusters.get(cid)?.topDocs.push([doc_id, count]);
        docIds.add(doc_id);
    }

    // DOC METADATA
    const docMeta: Record<string, DocMeta> = {};

    if (docIds.size > 0) {
        const ids = [...docIds];

        const rows = await execRows(
            `
            SELECT doc_id, author, title, pub_year
            FROM documents
            WHERE doc_id IN (${ ids.map(() => "?").join(",") })
            `,
            ids
        );

        for (const [id, author, title, pub_year] of rows as any[]) {
            docMeta[id] = { author, title, pub_year };
        }
    }

    // EXEMPLARS
    const clusterExemplars: Record<
        number,
        { event_id: string; doc_id: string; token_idx: number }[]
    > = {};

    const exemplarRows = await execRows(
        `
        SELECT e.cluster_id, e.event_id, e.doc_id, e.token_idx
        FROM events e
        ${ joins }
        ${ baseWhere }
        LIMIT 5000
        `,
        baseParams
    );

    for (const [cid, event_id, doc_id, token_idx] of exemplarRows as any[]) {
        if (!clusterExemplars[cid]) clusterExemplars[cid] = [];

        clusterExemplars[cid].push({
            event_id: String(event_id),
            doc_id,
            token_idx,
        });
    }

    // DIAGNOSTICS
    const multiClusterEvents = Object.entries(eventClusters)
        .filter(([_, c]) => c.length > 1)
        .map(([event_id, clusters]) => ({
            event_id,
            cluster_ids: clusters,
        }));

    const docClusterMap: Record<string, Set<number>> = {};

    for (const [event_id, clustersArr] of Object.entries(eventClusters)) {
        const doc_id = eventToDoc[event_id];
        if (!doc_id) continue;

        if (!docClusterMap[doc_id]) docClusterMap[doc_id] = new Set();

        for (const c of clustersArr) {
            docClusterMap[doc_id].add(c);
        }
    }

    const docClusterFootprint: Record<string, number[]> = {};

    for (const [doc, set] of Object.entries(docClusterMap)) {
        docClusterFootprint[doc] = Array.from(set);
    }

    const clusterStats = {
        totalEvents: Object.keys(eventClusters).length,
        multiClusterEvents: multiClusterEvents.length,
        multiClusterRate:
            Object.keys(eventClusters).length > 0
                ? multiClusterEvents.length / Object.keys(eventClusters).length
                : 0,
    };

    // console.log('docExemplars', clusterExemplars)

    return {
        concept,
        generated_at: new Date().toISOString(),
        clusters: [...clusters.values()],
        docMeta,
        docExemplars: clusterExemplars,

        diagnostics: {
            eventClusters,
            docClusterFootprint,
            clusterStats,
            multiClusterEvents,
        },
    };
}
