import { createSignal, Show } from "solid-js";

import type { ClusterAggregate, ClusterFile } from "./loadClusters";

import { getWindow } from "../../services/windowCache";
import { queryEventsByIds } from "../../services/db";
import type { ResolvedEvent } from "./ConceptClusters";

interface Props {
    clusters: ClusterFile;
    MAX_EXEMPLARS_PER_DOC: number;
}

export default function ClusterExport(props: Props) {
    const [exporting, setExporting] = createSignal(false);
    const [exportError, setExportError] = createSignal<string | null>(null);
    const [copyStatus, setCopyStatus] = createSignal<"idle" | "copied" | "failed">("idle");

    const buildExportPayload = async () => {
        const f = props.clusters;
        if (!f) return null;

        const allEventIds = Array.from(new Set(f.points.map((p) => p.event_id)));
        const eventById = await queryEventsByIds(allEventIds);

        const exemplarsByDoc = new Map<string, ResolvedEvent[]>();
        for (const p of f.points) {
            const ev = eventById.get(p.event_id);
            if (!ev) continue;
            const arr = exemplarsByDoc.get(ev.doc_id) ?? [];
            if (arr.length < props.MAX_EXEMPLARS_PER_DOC) {
                arr.push(ev);
                exemplarsByDoc.set(ev.doc_id, arr);
            }
        }

        const toExemplarPayload = (ev: ResolvedEvent) => ({
            event_id: ev.event_id,
            doc_id: ev.doc_id,
            token_idx: ev.token_idx,
            token: ev.token,
            window: getWindow(ev.event_id) ?? null,
        });

        const aggregates: Record<string, ClusterAggregate & {
            top_docs_resolved: {
                doc_id: string;
                count: number;
                exemplars: ReturnType<typeof toExemplarPayload>[];
            }[];
        }> = {};

        for (const [cid, agg] of Object.entries(f.clusters.aggregates)) {
            aggregates[cid] = {
                ...agg,
                top_docs_resolved: agg.top_docs.map(([doc_id, count]) => ({
                    doc_id,
                    count,
                    exemplars: (exemplarsByDoc.get(doc_id) ?? []).map(toExemplarPayload),
                })),
            };
        }

        return {
            type: f.type,
            concept: f.concept,
            generated_at: f.generated_at,
            n_events: f.n_events,
            bounds: f.bounds,
            globalBounds: f.globalBounds,
            clusters: {
                label_map: f.clusters.label_map,
                aggregates,
            },
            points: f.points,
            exported_at: new Date().toISOString(),
        };
    };

    const exportFilename = () => {
        const f = props.clusters;
        const concept = f?.concept ?? "cluster_export";
        return `${ concept }_clusters_export.json`;
    };

    const handleCopyJson = async () => {
        setExportError(null);
        setCopyStatus("idle");
        setExporting(true);
        try {
            const payload = await buildExportPayload();
            if (!payload) return;
            const text = JSON.stringify(payload, null, 2);
            await navigator.clipboard.writeText(text);
            setCopyStatus("copied");
            setTimeout(() => setCopyStatus("idle"), 2000);
        } catch (err) {
            console.error(err);
            setCopyStatus("failed");
            setExportError(err instanceof Error ? err.message : "Failed to copy to clipboard");
            setTimeout(() => setCopyStatus("idle"), 2000);
        } finally {
            setExporting(false);
        }
    };

    const handleDownloadJson = async () => {
        setExportError(null);
        setExporting(true);
        try {
            const payload = await buildExportPayload();
            if (!payload) return;
            const text = JSON.stringify(payload, null, 2);
            const blob = new Blob([text], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = exportFilename();
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error(err);
            setExportError(err instanceof Error ? err.message : "Failed to build download");
        } finally {
            setExporting(false);
        }
    };


    return (
        <>
            <button class="border small" disabled={exporting()} onClick={handleCopyJson} >
                <i>content_copy</i>
                <span>
                    {copyStatus() === "copied" ? "Copied!" : copyStatus() === "failed"
                        ? "Copy failed" : "Copy JSON"}
                </span>
            </button>

            <button class="border small" disabled={exporting()} onClick={handleDownloadJson} >
                <i>download</i>
                <span>Download JSON</span>
            </button>

            <Show when={exporting()}>
                <progress class="circle small" />
            </Show>

            <Show when={exportError()}>
                <span class="error-container">{exportError()}</span>
            </Show>
        </>
    );
}