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
        string,
        {
            event_id: string;
            doc_id: string;
            token_idx: number;
        }[]
    >;
}

function yearFilter(yearMode: YearMode, fromYear: number, toYear: number): string {
    return yearMode === "single"
        ? `AND pub_year = ${ fromYear }`
        : `AND pub_year BETWEEN ${ fromYear } AND ${ toYear }`;
}

interface loadClusterReportParams {
    concept: string;
    yearMode?: YearMode;
    fromYear?: number;
    toYear?: number;
}

export async function loadClusterReport(params: loadClusterReportParams): Promise<ClusterReport> {
    const { concept, yearMode, fromYear, toYear } = params;

    const yearActive = yearMode != null && fromYear != null && toYear != null;
    const yearClause = yearActive ? yearFilter(yearMode!, fromYear!, toYear!) : "";

    // base stats label, size
    const clusterRows = await execRows(
        `
        SELECT cluster_id, cluster_label, COUNT(*) AS n
        FROM events
        WHERE concept = ?
          AND cluster_id IS NOT NULL
          AND pub_year IS NOT NULL
          ${ yearClause }
        GROUP BY cluster_id, cluster_label
        `,
        [concept]
    );

    const clusters = new Map<number, ClusterSummary>();

    for (const [id, label, n] of clusterRows as any[]) {
        clusters.set(id, {
            id,
            label,
            eventCount: n,
            topTokens: [],
            topDocs: []
        });
    }

    if (clusters.size === 0) {
        return {
            concept,
            generated_at: new Date().toISOString(),
            clusters: [],
            docMeta: {},
            docExemplars: {}
        };
    }

    // tokens
    const tokenRows = await execRows(
        `
        SELECT cluster_id, token, c
        FROM (
            SELECT
                cluster_id,
                token,
                COUNT(*) AS c,
                ROW_NUMBER() OVER (
                    PARTITION BY cluster_id
                    ORDER BY COUNT(*) DESC
                ) AS rn
            FROM events
            WHERE concept = ?
              AND cluster_id IS NOT NULL
              AND pub_year IS NOT NULL
              ${ yearClause }
            GROUP BY cluster_id, token
        )
        WHERE rn <= 25
        `,
        [concept]
    );

    for (const [cid, token, c] of tokenRows as any[]) {
        clusters.get(cid)?.topTokens.push([token, c]);
    }

    // top docs per cluster
    const docRows = await execRows(
        `
        SELECT cluster_id, doc_id, c
        FROM (
            SELECT
                cluster_id,
                doc_id,
                COUNT(*) AS c,
                ROW_NUMBER() OVER (
                    PARTITION BY cluster_id
                    ORDER BY COUNT(*) DESC
                ) AS rn
            FROM events
            WHERE concept = ?
              AND cluster_id IS NOT NULL
              AND pub_year IS NOT NULL
              ${ yearClause }
            GROUP BY cluster_id, doc_id
        )
        WHERE rn <= 25
        `,
        [concept]
    );

    const docIds = new Set<string>();

    for (const [cid, doc_id, c] of docRows as any[]) {
        clusters.get(cid)?.topDocs.push([doc_id, c]);
        docIds.add(doc_id);
    }

    // metadata
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

    // exemplars
    const docExemplars: Record<
        string,
        {
            event_id: string;
            doc_id: string;
            token_idx: number;
        }[]
    > = {};

    if (docIds.size > 0) {
        const ids = [...docIds];

        const exemplarRows = await execRows(
            `
            SELECT event_id, doc_id, token_idx
            FROM events
            WHERE concept = ?
              AND cluster_id IS NOT NULL
              AND pub_year IS NOT NULL
              ${ yearClause }
              AND doc_id IN (${ ids.map(() => "?").join(",") })
            LIMIT 5000
            `,
            [concept, ...ids]
        );

        for (const [event_id, doc_id, token_idx] of exemplarRows as any[]) {
            if (!docExemplars[doc_id]) {
                docExemplars[doc_id] = [];
            }

            docExemplars[doc_id].push({
                event_id: String(event_id),
                doc_id,
                token_idx
            });
        }
    }

    return {
        concept,
        generated_at: new Date().toISOString(),
        clusters: [...clusters.values()],
        docMeta,
        docExemplars
    };
}
