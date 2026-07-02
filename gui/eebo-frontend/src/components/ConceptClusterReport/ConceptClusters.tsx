import { createSignal, createEffect, createMemo, Show, For, createResource } from "solid-js";

import { controls } from "../../state/controls.store";
import { queryEventById, queryEventsByIds } from "../../services/db";
import ControlsHeader from "../ControlsHeader";

import { loadClusters } from "./loadClusters";
import DocRow from "./DocRow";
import ClusterExport from "./ClusterExport";

interface ClusterAggregate {
    top_tokens: [string, number][];
    top_docs: [string, number][];
}

export interface ClusterInfo { // a point without point data
    event_id: string;
    cluster_id: number;
    cluster_label: string | null;
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
    points: ClusterInfo[];
}

export type ResolvedEvent = NonNullable<Awaited<ReturnType<typeof queryEventById>>>;

const CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#BA7517", "#639922", "#E24B4A",
];

const MAX_EXEMPLARS_PER_DOC = 3;

// Resource source: returns null (skips fetch) when no concept selected,
// otherwise the params object. createResource refetches whenever any
// field read here changes, which is what replaces the two createEffects.
function clusterFetchParams() {
    const concept = controls.concept;
    if (!concept) return null;
    return {
        concept,
        yearMode: controls.yearMode,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
    };
}

export default function ConceptClusters() {
    const [clusterFile, { refetch: refetchClusters }] = createResource(
        clusterFetchParams,
        async (params): Promise<ClusterFile> => loadClusters(params)
    );

    const [selectedCluster, setSelectedCluster] = createSignal<string | null>(null);
    const [showDominantOnly, setShowDominantOnly] = createSignal(true);

    // Reset cluster selection whenever a fresh file loads (new concept or
    // year range) — createResource doesn't clear selectedCluster itself,
    // so do it explicitly when the underlying data identity changes.
    createEffect(() => {
        const f = clusterFile();
        if (!f) {
            setSelectedCluster(null);
            return;
        }
        const first = Object.keys(f.clusters.aggregates)[0];
        setSelectedCluster(first ?? null);
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

    const visibleClusterSummary = createMemo(() => {
        const summary = clusterSummary();

        if (!showDominantOnly() || summary.length === 0) {
            return summary;
        }

        const largest = Math.max(...summary.map(c => c.n));
        const cutoff = largest * 0.10;

        return summary.filter(c => c.n > cutoff);
    });

    createEffect(() => {
        const visible = visibleClusterSummary();

        if (!visible.length) return;

        const current = selectedCluster();

        if (!visible.some(c => c.cid === current)) {
            setSelectedCluster(visible[0].cid);
        }
    });

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

    const clusterPoints = () => {
        const f = clusterFile();
        const cid = selectedCluster();
        if (!f || cid === null) return [];
        return f.points.filter(p => String(p.cluster_id) === cid);
    };

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


    return (
        <article class="concept-clusters background">
            <ControlsHeader>
                <Show when={clusterFile()}>
                    <ClusterExport MAX_EXEMPLARS_PER_DOC={MAX_EXEMPLARS_PER_DOC} clusters={clusterFile()!} />
                </Show>
            </ControlsHeader>

            <Show when={!controls.concept}>
                <p>Select a concept to view its clusters.</p>
            </Show>

            {/* <Show when={clusterFile.loading}>
                <p>Loading clusters</p>
                <progress />
            </Show> */}

            <Show when={clusterFile.error}>
                <aside class="error-container">
                    <h3>Error</h3>
                    {clusterFile.error instanceof Error ? clusterFile.error.message : String(clusterFile.error)}
                </aside>
            </Show>

            <Show when={!clusterFile.loading && (!clusterFile() || !clusterFile()?.n_events)}>
                <h3 class="center-align middle-align  extra-margin extra-padding">No data for this period.</h3>
            </Show>

            <h2>UMAP/PACMAP Report</h2>
            <Show when={clusterFile() && clusterFile()?.n_events}>
                <nav class="scroll bottom-padding">
                    <span> Dominant clusters only </span>
                    <div class="field middle-align">
                        <label class="switch">
                            <input
                                type="checkbox"
                                checked={showDominantOnly()}
                                onInput={(e) => setShowDominantOnly(e.currentTarget.checked)}
                            />
                            <span></span>
                        </label>
                    </div>

                    <For each={visibleClusterSummary()}>
                        {(row, i) => (
                            <button class="chip"
                                onClick={() => setSelectedCluster(row.cid)}
                                style={{
                                    "background": selectedCluster() === row.cid ? CLUSTER_COLORS[i() % CLUSTER_COLORS.length] : "var(--color-background-secondary)",
                                    "border": `1pt solid ${ CLUSTER_COLORS[i() % CLUSTER_COLORS.length] }`,
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
                </nav>

                {/* Selected cluster detail */}
                <Show when={selectedAgg()}>
                    <div class="grid">
                        <div class="s3">
                            <section class="left-padding">
                                <div class="large-height scroll surface">

                                    <table class="stripes no-border scroll max">
                                        <caption>Top Tokens</caption>
                                        <thead class="fixed">
                                            <tr><th>Rank</th><th>Token</th><th>Count</th></tr>
                                        </thead>
                                        <tbody>
                                            <For each={selectedAgg()!.top_tokens}>
                                                {([token, count], i) => (
                                                    <tr>
                                                        <td>{i() + 1}</td>
                                                        <td><strong>{token}</strong></td>
                                                        <td>{count.toLocaleString()}</td>
                                                    </tr>
                                                )}
                                            </For>
                                        </tbody>
                                    </table>

                                    <Show when={clusterFile()}>
                                        <p>
                                            {clusterFile()!.n_events.toLocaleString()} events
                                        </p><p>
                                            {Object.keys(clusterFile()!.clusters.aggregates).length} clusters
                                        </p><p>
                                            {noiseCount().toLocaleString()} noise points
                                        </p><p>
                                            Generated {clusterFile()!.generated_at.slice(0, 10)}
                                        </p>
                                    </Show>

                                </div>
                            </section>
                        </div>

                        <div class="s9">
                            <section class="scroll-parent" style={{ height: '70vh' }}>
                                <Show when={clusterEvents.loading}>
                                    <progress />
                                </Show>
                                <div class="surface" style={{ overflow: "auto" }}>
                                    <table class="stripes no-border scroll max">
                                        <caption>Top documents</caption>
                                        <thead class="fixed surface-container-high">
                                            <tr><th>Rank</th><th>Document ID</th><th>Count</th></tr>
                                        </thead>
                                        <tbody>
                                            <For each={selectedAgg()!.top_docs}>
                                                {([doc_id, count], i) => (
                                                    <DocRow
                                                        rank={i() + 1}
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

        </article >
    );
}