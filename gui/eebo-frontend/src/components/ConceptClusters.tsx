import { createSignal, createEffect, createMemo, Show, For, createResource, onCleanup } from "solid-js";
import { queryEventById } from "../services/db";
import { controls, setControls } from "../state/controls.store";
import { fetchWindow } from "../services/tokenWindowApi";
import { loadJson } from "../lib/loadJson";
import ControlsHeader from "./ControlsHeader";

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

const CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#BA7517", "#639922", "#E24B4A",
];

export default function ConceptClusters() {
    const [clusterFile, setClusterFile] = createSignal<ClusterFile | null>(null);
    const [selectedCluster, setSelectedCluster] = createSignal<string | null>(null);
    const [clusterLoading, setClusterLoading] = createSignal(false);
    const [clusterError, setClusterError] = createSignal<string | null>(null);

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
                        <div class="s6">
                            <section>
                                <h3>Top tokens — cluster {clusterFile()!.clusters.label_map[selectedCluster()!]}</h3>
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

                        {/* Top documents — exemplars + window text rendered under each doc */}
                        <div class="s6">
                            <section>
                                <h3>Top documents — cluster {clusterFile()!.clusters.label_map[selectedCluster()!]}</h3>
                                <div class="large-height scroll surface">
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
                                                        points={clusterPoints()}
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
    points: ClusterPoint[];
}) {
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

    const [resolved] = createResource(
        () => (visible() ? props.points.map(p => p.event_id) : null),
        async (eventIds) => {
            const events = await Promise.all(
                eventIds.map(id => queryEventById(id))
            );
            return events.filter((e): e is NonNullable<typeof e> => e !== null);
        }
    );

    const exemplarsForDoc = createMemo(() => {
        const events = resolved();
        if (!events) return [];
        return events
            .filter(e => e.doc_id === props.doc_id)
            .slice(0, 3);
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
            <For each={exemplarsForDoc()}>
                {(ev) => <ExemplarSubRow event={ev} />}
            </For>
        </>
    );
}



function ExemplarSubRow(props: { event: NonNullable<Awaited<ReturnType<typeof queryEventById>>> }) {
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

    const [windowText] = createResource(
        () => (visible() ? { doc_id: props.event.doc_id, token_idx: props.event.token_idx } : null),
        (e) => fetchWindow(e)
    );

    return (
        <tr
            ref={rowRef}
            onClick={() => setControls("selectedEventId", props.event.event_id)}
            style={{
                background: controls.selectedEventId === props.event.event_id
                    ? "var(--color-background-info)"
                    : "transparent",
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
                    <Show when={visible() && windowText.loading}>
                        <progress />
                    </Show>
                    <Show when={windowText.error}>
                        <span class="error-container">Failed to load window: {String(windowText.error)}</span>
                    </Show>
                    <Show when={windowText() && !windowText.loading}>
                        <span innerHTML={windowText()} />
                    </Show>
                </div>
            </td>
        </tr>
    );
}