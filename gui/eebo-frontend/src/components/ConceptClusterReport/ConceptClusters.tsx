import { createSignal, createMemo, Show, For, createResource, createEffect } from "solid-js";
import { controls } from "../../state/controls.store";
import ControlsHeader from "../ControlsHeader";

import { loadClusterReport, type ClusterReport } from "./loadClusterReport";
import DocRow from "./DocRow";
import ClusterExport from "./ClusterExport";

const CLUSTER_COLORS = [
    "#7F77DD", "#1D9E75", "#D85A30", "#D4537E",
    "#378ADD", "#BA7517", "#639922", "#E24B4A",
];

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
    const [clusterReport] = createResource(
        clusterFetchParams,
        loadClusterReport
    );

    const [selectedCluster, setSelectedCluster] = createSignal<number | null>(null);
    const [showDominantOnly, setShowDominantOnly] = createSignal(true);

    const clusters = () => clusterReport()?.clusters ?? [];

    const visibleClusters = createMemo(() => {
        const list = clusters();
        if (!showDominantOnly() || list.length === 0) return list;
        const max = Math.max(...list.map(c => c.eventCount));
        const cutoff = max * 0.1;
        return list.filter(c => c.eventCount > cutoff);
    });

    const selectedClusterData = createMemo(() => clusters().find(c => c.id === selectedCluster()) ?? null);

    createEffect(() => {
        const report = clusterReport();
        if (!report?.clusters.length) {
            setSelectedCluster(null);
            return;
        }
        setSelectedCluster(report.clusters[0].id);
    });

    return (
        <article class="concept-clusters background">

            <ControlsHeader>
                <Show when={clusterReport()}>
                    <ClusterExport clusters={clusterReport()!} />
                </Show>
            </ControlsHeader>

            <Show when={!controls.concept}>
                <p>Select a concept to view clusters.</p>
            </Show>

            <Show when={clusterReport.error}>
                <aside class="error-container">
                    {String(clusterReport.error)}
                </aside>
            </Show>

            <Show when={clusterReport()}>
                <h2>Concept Clusters</h2>

                <nav class="scroll bottom-padding">
                    <div class="field middle-align">
                        <label class="switch">
                            <span>Dominant clusters only</span>
                            <input class="left-margin"
                                type="checkbox"
                                checked={showDominantOnly()}
                                onInput={e => setShowDominantOnly(e.currentTarget.checked)}
                            />
                            <span></span>
                        </label>
                    </div>

                    <For each={visibleClusters()}>
                        {(c, i) => (
                            <button
                                class="chip"
                                onClick={() => setSelectedCluster(c.id)}
                                style={{
                                    background:
                                        selectedCluster() === c.id
                                            ? CLUSTER_COLORS[i() % CLUSTER_COLORS.length]
                                            : "var(--color-background-secondary)"
                                }}
                            >
                                <strong>{c.label ?? c.id}</strong>
                                <span>
                                    {c.eventCount} · {c.topTokens[0]?.[0] ?? "—"}
                                </span>
                            </button>
                        )}
                    </For>
                </nav>

                <Show when={selectedClusterData()}>
                    {(c) => (
                        <div class="grid">

                            <section class="s2">
                                <table class="scroll small-height">
                                    {/* <caption>Top Tokens</caption> */}
                                    <thead class="fixed surface-container-highest">
                                        <tr>
                                            <td></td>
                                            <td>Token</td>
                                            <td>Count</td>
                                        </tr>
                                    </thead>
                                    <For each={c().topTokens}>
                                        {([t, n], i) => (
                                            <tr>
                                                <td>{i() + 1}</td>
                                                <td>{t}</td>
                                                <td>{n}</td>
                                            </tr>
                                        )}
                                    </For>
                                </table>
                            </section>

                            <section class="s10">
                                <table class="scroll small-height">
                                    {/* <caption>Top Documents</caption> */}
                                    {/* <thead class="fixed">
                                        <tr>
                                            <td></td>
                                            <td></td>
                                            <td></td>
                                            <td></td>
                                            <td></td>
                                        </tr>
                                    </thead> */}
                                    <tbody>
                                        <For each={c().topDocs}>
                                            {([doc_id, count], i) => (
                                                <DocRow
                                                    rank={i() + 1}
                                                    doc_id={doc_id}
                                                    count={count}
                                                    author={clusterReport()?.docMeta[doc_id]?.author ?? null}
                                                    pub_year={clusterReport()?.docMeta[doc_id]?.pub_year ?? null}
                                                    title={clusterReport()?.docMeta[doc_id]?.title ?? null}
                                                    exemplars={clusterReport()?.docExemplars?.[doc_id] ?? []}
                                                />
                                            )}
                                        </For>
                                    </tbody>
                                </table>
                            </section>

                        </div>
                    )}
                </Show>
            </Show>
        </article>
    );
}