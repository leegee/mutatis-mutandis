import { createSignal, createEffect, createMemo, Show, For, createResource, onCleanup } from "solid-js";
import { controls, setControls } from "../state/controls.store";
import { queryEventById, queryEventsByIds } from "../services/db";
import { fetchWindowBatch, type TextWindowItem } from "../services/tokenWindowBatchApi";
import { setWindowCache, getWindow } from "../services/windowCache";
import { loadJson } from "../lib/json";
import ControlsHeader from "./ControlsHeader";
import { controlsActions } from "../state/controls.actions";

interface ClusterAggregate {
    top_tokens: [string, number][];
    top_docs: [string, number][];
}

interface ClusterPoint {
    event_id: string;
    cluster_id: number;
    cluster_label: string | null;
    x: number;
    y: number;
    nx: number;
    ny: number;
    gx: number;
    gy: number;
    gnx: number;
    gny: number;
}

interface ClusterFile {
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

type ResolvedEvent = NonNullable<Awaited<ReturnType<typeof queryEventById>>>;

const CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#BA7517", "#639922", "#E24B4A",
];

// Max exemplar rows shown per document
const MAX_EXEMPLARS_PER_DOC = 3;

export default function ConceptClusters() {
    const [clusterFile, setClusterFile] = createSignal<ClusterFile | null>(null);
    const [selectedCluster, setSelectedCluster] = createSignal<string | null>(null);
    const [clusterLoading, setClusterLoading] = createSignal(false);
    const [clusterError, setClusterError] = createSignal<string | null>(null);

    const [exporting, setExporting] = createSignal(false);
    const [exportError, setExportError] = createSignal<string | null>(null);
    const [copyStatus, setCopyStatus] = createSignal<"idle" | "copied" | "failed">("idle");

    const runClusterFetch = async (concept: string) => {
        setClusterFile(null);
        setSelectedCluster(null);
        setClusterError(null);

        if (!concept) return;

        setClusterLoading(true);
        try {
            const data = await loadJson(`/data/scatter/concept_clusters/${ concept }.json`);
            setClusterFile(data);
            const first = Object.keys(data.clusters.aggregates)[0];
            setSelectedCluster(first ?? null);
        } catch (err) {
            console.error(err);
            setClusterError(err instanceof Error ? err.message : "Failed to load cluster file");
        } finally {
            setClusterLoading(false);
        }
    };

    createEffect(() => {
        const currentConcept = controls.concept;
        if (currentConcept) {
            runClusterFetch(currentConcept);
        } else {
            setClusterFile(null);
            setSelectedCluster(null);
            setClusterError(null);
        }
    });

    const clusterSummary = () => {
        const f = clusterFile();
        if (!f) return [];
        const { label_map, aggregates } = f.clusters;
        const pointsPerCluster = f.points.reduce(
            (acc, p) => {
                if (p.cluster_id !== -1) acc[p.cluster_id] = (acc[p.cluster_id] ?? 0) + 1;
                return acc;
            },
            {} as Record<number, number>
        );
        return Object.entries(aggregates).map(([cid, agg]) => ({
            cid,
            label: label_map[cid] ?? cid,
            n: pointsPerCluster[+cid] ?? 0,
            topToken: agg.top_tokens[0]?.[0] ?? "—",
        }));
    };

    const selectedAgg = () => {
        const f = clusterFile();
        const cid = selectedCluster();
        if (!f || cid === null) return null;
        return f.clusters.aggregates[cid] ?? null;
    };

    const noiseCount = () => {
        const f = clusterFile();
        if (!f) return 0;
        return f.points.filter(p => p.cluster_id === -1).length;
    };

    // All points belonging to the selected cluster
    const clusterPoints = () => {
        const f = clusterFile();
        const cid = selectedCluster();
        if (!f || cid === null) return [];
        return f.points.filter(p => String(p.cluster_id) === cid);
    };

    // Resolve every event in the selected cluster in ONE batched query (not
    // per DocRow, not one-by-one). This is the single source of `doc_id`
    // for each event_id, since ClusterPoint itself doesn't carry doc_id.
    const [clusterEvents] = createResource(
        clusterPoints,
        async (points): Promise<ResolvedEvent[]> => {
            if (!points.length) return [];
            const ids = points.map((p) => p.event_id);
            const eventMap = await queryEventsByIds(ids);
            const result: ResolvedEvent[] = [];
            for (const id of ids) {
                const ev = eventMap.get(id);
                if (ev) result.push(ev);
            }
            return result;
        }
    );

    // Group resolved events by doc_id, capped to MAX_EXEMPLARS_PER_DOC per
    // doc up front — so DocRow never has to fetch/filter the full cluster.
    const eventsByDoc = createMemo(() => {
        const events = clusterEvents();
        const map = new Map<string, ResolvedEvent[]>();
        if (!events) return map;
        for (const e of events) {
            const arr = map.get(e.doc_id) ?? [];
            if (arr.length < MAX_EXEMPLARS_PER_DOC) {
                arr.push(e);
                map.set(e.doc_id, arr);
            }
        }
        return map;
    });

    // Build the full export payload: the loaded cluster file, with every
    // cluster's top_docs annotated with resolved exemplar events (capped
    // to MAX_EXEMPLARS_PER_DOC per doc, same as the live view).
    //
    // This resolves events for ALL clusters in the file, not just the
    // currently selected one, since the export is meant to be a complete
    // snapshot of "the whole cluster file with doc/exemplars resolved".
    const buildExportPayload = async () => {
        const f = clusterFile();
        if (!f) return null;

        // Resolve every event referenced anywhere in f.points, in one
        // batched query, regardless of which cluster it belongs to.
        const allEventIds = Array.from(new Set(f.points.map((p) => p.event_id)));
        const eventById = await queryEventsByIds(allEventIds);

        // doc_id -> exemplar events, capped per doc, across the whole file
        const exemplarsByDoc = new Map<string, ResolvedEvent[]>();
        for (const p of f.points) {
            const ev = eventById.get(p.event_id);
            if (!ev) continue;
            const arr = exemplarsByDoc.get(ev.doc_id) ?? [];
            if (arr.length < MAX_EXEMPLARS_PER_DOC) {
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

        // Rebuild aggregates with resolved exemplars attached to each top_doc
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
        const f = clusterFile();
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
        <article class="concept-clusters">
            <ControlsHeader />

            <h2>Cluster Explorer</h2>

            <Show when={!controls.concept}>
                <p>Select a concept to view its clusters.</p>
            </Show>

            <Show when={clusterLoading()}>
                <p>Loading clusters…</p>
            </Show>

            <Show when={clusterError()}>
                <aside class="error"><h3>Error</h3>{clusterError()}</aside>
            </Show>

            <Show when={clusterFile()}>
                <p>
                    {clusterFile()!.n_events.toLocaleString()} events
                    · {Object.keys(clusterFile()!.clusters.aggregates).length} clusters
                    · {noiseCount().toLocaleString()} noise points
                    · generated {clusterFile()!.generated_at.slice(0, 10)}
                </p>

                {/* Export controls */}
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:1rem;">
                    <button
                        class="border small"
                        disabled={exporting()}
                        onClick={handleCopyJson}
                    >
                        <i>content_copy</i>
                        <span>
                            {copyStatus() === "copied"
                                ? "Copied!"
                                : copyStatus() === "failed"
                                    ? "Copy failed"
                                    : "Copy JSON"}
                        </span>
                    </button>

                    <button
                        class="border small"
                        disabled={exporting()}
                        onClick={handleDownloadJson}
                    >
                        <i>download</i>
                        <span>Download JSON</span>
                    </button>

                    <Show when={exporting()}>
                        <progress class="circle small" />
                    </Show>

                    <Show when={exportError()}>
                        <span class="error-text">{exportError()}</span>
                    </Show>
                </div>

                {/* Cluster selector pills */}
                <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1.5rem;">
                    <For each={clusterSummary()}>
                        {(row, i) => (
                            <button class="chip"
                                onClick={() => setSelectedCluster(row.cid)}
                                style={{
                                    "background": selectedCluster() === row.cid
                                        ? CLUSTER_COLORS[i() % CLUSTER_COLORS.length]
                                        : "var(--color-background-secondary)",
                                    "color": selectedCluster() === row.cid
                                        ? "#fff"
                                        : "var(--color-text-primary)",
                                }}
                            >
                                <strong>{row.label}</strong>
                                <span class="medium-opacity">
                                    {row.n.toLocaleString()} · {row.topToken}
                                </span>
                            </button>
                        )}
                    </For>
                </div>

                {/* Selected cluster detail */}
                <Show when={selectedAgg()}>
                    <div class="grid">
                        <div class="s3">
                            <section>
                                <h3>Top tokens</h3>
                                <div class="large-height scroll surface">
                                    <table class="stripes no-border scroll max">
                                        <thead class="fixed">
                                            <tr><th>Rank</th><th>Token</th><th>Count</th></tr>
                                        </thead>
                                        <tbody>
                                            <For each={selectedAgg()!.top_tokens}>
                                                {([token, count], i) => (
                                                    <tr>
                                                        <td>{i()}</td>
                                                        <td><strong>{token}</strong></td>
                                                        <td>{count.toLocaleString()}</td>
                                                    </tr>
                                                )}
                                            </For>
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </div>

                        <div class="s9">
                            <section class="scroll-parent" style={{ height: '70vh' }}>
                                <h3>Top documents</h3>
                                <Show when={clusterEvents.loading}>
                                    <progress />
                                </Show>
                                <div class="surface" style={{ overflow: "auto" }}>
                                    <table class="stripes no-border scroll max">
                                        <thead class="fixed">
                                            <tr><th>Rank</th><th>Document ID</th><th>Count</th></tr>
                                        </thead>
                                        <tbody>
                                            <For each={selectedAgg()!.top_docs}>
                                                {([doc_id, count], i) => (
                                                    <DocRow
                                                        rank={i()}
                                                        doc_id={doc_id}
                                                        count={count}
                                                        events={eventsByDoc().get(doc_id) ?? []}
                                                    />
                                                )}
                                            </For>
                                        </tbody>
                                    </table>
                                </div>
                            </section>
                        </div>
                    </div>
                </Show>
            </Show>
        </article>
    );
}

function DocRow(props: {
    rank: number;
    doc_id: string;
    count: number;
    events: ResolvedEvent[];
}) {
    let rowRef: HTMLTableRowElement | undefined;
    const [visible, setVisible] = createSignal(false);

    // events are pre-resolved and already capped to MAX_EXEMPLARS_PER_DOC
    // by the parent — only fetch window text for these few ids, and only
    // once this row is visible.
    const [resolved] = createResource(
        () => (visible() ? props.events : null),
        async (events): Promise<ResolvedEvent[]> => {
            if (!events || !events.length) return [];

            // skip ids whose window content is already cached
            const toFetch = events.filter((e) => !getWindow(e.event_id));

            if (toFetch.length) {
                const batch = toFetch.map((e) => ({
                    eventId: e.event_id,
                    docId: e.doc_id,
                    tokenIdx: e.token_idx,
                }));

                const res = await fetchWindowBatch(batch);

                res.results.forEach((r: TextWindowItem, idx: number) => {
                    setWindowCache(toFetch[idx].event_id, r.content);
                });
            }

            return events;
        }
    );

    createEffect(() => {
        if (!rowRef || visible()) return;
        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting) {
                    setVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "100px" }
        );
        observer.observe(rowRef);
        onCleanup(() => observer.disconnect());
    });

    return (
        <>
            <tr ref={rowRef}>
                <td>{props.rank}</td>
                <td><strong>{props.doc_id}</strong></td>
                <td>{props.count.toLocaleString()}</td>
            </tr>
            <Show when={visible() && resolved.loading}>
                <tr>
                    <td colspan="3">
                        <progress />
                    </td>
                </tr>
            </Show>
            <For each={props.events}>
                {(ev) => <ExemplarSubRow event={ev} />}
            </For>
        </>
    );
}


function ExemplarSubRow(props: { event: ResolvedEvent }) {
    let rowRef: HTMLTableRowElement | undefined;
    const [visible, setVisible] = createSignal(false);

    createEffect(() => {
        if (!rowRef || visible()) return;

        const observer = new IntersectionObserver(
            (entries) => {
                if (entries[0].isIntersecting) {
                    setVisible(true);
                    observer.disconnect();
                }
            },
            { rootMargin: "100px" }
        );

        observer.observe(rowRef);
        onCleanup(() => observer.disconnect());
    });

    const windowText = createMemo(() => getWindow(props.event.event_id));

    return (
        <tr
            ref={rowRef}
            onClick={() => controlsActions.setSelectedEventIds(props.event.event_id)}
            class="bottom-padding"
            style={{
                background:
                    controls.selectedEventId === props.event.event_id
                        ? "var(--color-background-info)" : "transparent",
            }}
        >
            <td colspan="3">
                <div>
                    token_idx {props.event.token_idx} · {props.event.token}
                </div>

                <div>
                    <Show when={!visible()}>
                        &mdash;
                    </Show>

                    <Show when={visible() && !windowText()}>
                        <progress />
                    </Show>

                    <Show when={windowText()}>
                        <span innerHTML={windowText()!} />
                    </Show>
                </div>
            </td>
        </tr>
    );
}
