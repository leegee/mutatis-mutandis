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
): Promise<ClusterReport> {
    const { concept, yearMode, fromYear, toYear, authorMatch } = params;

    const yearActive =
        yearMode != null && fromYear != null && toYear != null;

    const year =
        yearActive && yearMode && fromYear != null && toYear != null
            ? yearFilter(yearMode, fromYear, toYear)
            : null;

    const author = authorMatch?.trim()
        ? authorFilter(authorMatch)
        : null;

    const joins = author ? "JOIN documents d ON d.doc_id = e.doc_id" : "";
    const authorSql = author ? author.sql : "";

    const baseWhere = `
        WHERE e.concept = ?
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

    // cluster stats
    const clusterRows = await execRows(
        `
        SELECT e.cluster_id, e.cluster_label, COUNT(*) AS n
        FROM events e
        ${ joins }
        ${ baseWhere }
        GROUP BY e.cluster_id, e.cluster_label
        `,
        baseParams
    );

    const clusters = new Map<number, ClusterSummary>();

    for (const [id, label, n] of clusterRows as any[]) {
        clusters.set(id, {
            id,
            label,
            eventCount: n,
            topTokens: [],
            topDocs: [],
        });
    }

    if (clusters.size === 0) {
        return {
            concept,
            generated_at: new Date().toISOString(),
            clusters: [],
            docMeta: {},
            docExemplars: {},
        };
    }

    // tokens
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

    // top docs
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
    const docExemplars: ClusterReport["docExemplars"] = {};

    if (docIds.size > 0) {
        const ids = [...docIds];

        const exemplarRows = await execRows(
            `
            SELECT e.event_id, e.doc_id, e.token_idx
            FROM events e
            ${ joins }
            ${ baseWhere }
              AND e.doc_id IN (${ ids.map(() => "?").join(",") })
            LIMIT 5000
            `,
            [...baseParams, ...ids]
        );

        for (const [event_id, doc_id, token_idx] of exemplarRows as any[]) {
            if (!docExemplars[doc_id]) docExemplars[doc_id] = [];

            docExemplars[doc_id].push({
                event_id: String(event_id),
                doc_id,
                token_idx,
            });
        }
    }

    return {
        concept,
        generated_at: new Date().toISOString(),
        clusters: [...clusters.values()],
        docMeta,
        docExemplars,
    };
}