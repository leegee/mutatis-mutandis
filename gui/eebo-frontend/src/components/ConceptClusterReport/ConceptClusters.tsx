import { createSignal, createMemo, Show, For, createResource, createEffect } from "solid-js";
import { controls } from "../../state/controls.store";
import ControlsHeader from "../ControlsHeader";

import { loadClusterReport } from "./loadClusterReport";
import DocRow from "./DocRow";
import ClusterExport from "./ClusterExport";

import "./ConceptClusters.css";
import { buildCssColorMap, type CssClusterColor } from "../../lib/colour";

function clusterFetchParams() {
    const concept = controls.concept;
    if (!concept) return null;

    return {
        concept,
        yearMode: controls.yearMode,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        authorMatch: controls.authorMatch,
    };
}

export default function ConceptClusters() {
    const [clusterReport] = createResource(
        clusterFetchParams,
        loadClusterReport
    );

    const [selectedCluster, setSelectedCluster] = createSignal<number | null>(null);
    const [showDominantOnly, setShowDominantOnly] = createSignal(true);
    const [showExemplars, setShowExemplars] = createSignal(false);

    const clusters = () => clusterReport()?.clusters ?? [];

    const clusterColors = createMemo(() =>
        buildCssColorMap(
            clusters()
                .sort((a, b) => b.eventCount - a.eventCount)
                .map(c => String(c.id)),
            24
        )
    );

    const getClusterColor = (id: number): CssClusterColor =>
        clusterColors().get(String(id)) ?? {
            bg: "rgb(128,128,128)",
            fg: "#fff"
        };

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

    const clusterExemplarsByDoc = createMemo(() => {
        const report = clusterReport();
        const cid = selectedCluster();
        if (!report || cid == null) return {};

        const rows = report.docExemplars?.[cid] ?? [];

        const map: Record<string, typeof rows> = {};

        for (const r of rows) {
            if (!map[r.doc_id]) map[r.doc_id] = [];
            map[r.doc_id].push(r);
        }

        return map;
    });

    return (
        <article class="concept-clusters background">
            <ControlsHeader authorMatch>
                <Show when={clusterReport()}>
                    <ClusterExport clusters={clusterReport()!} />
                </Show>

                <label class="switch">
                    <input type="checkbox"
                        checked={showExemplars()}
                        onInput={e => setShowExemplars(e.currentTarget.checked)}
                    />
                    <span></span>
                </label>
                <span>Show exemplars</span>
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
                <h2 class="max"> Concept Clusters </h2>

                <nav class="scroll bottom-padding">
                    <div class="field middle-align">
                        <label class="switch">
                            <span class="small-text">Dominant <br />clusters only</span>
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
                                    background: selectedCluster() === c.id ? getClusterColor(c.id).bg : "var(--color-background-secondary)",
                                    color: selectedCluster() === c.id ? getClusterColor(c.id).fg : "var(--color-secondary)",
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
                            <div class="s2">
                                <table class="surface">
                                    <caption>Top Tokens</caption>
                                    <For each={c().topTokens}>
                                        {([t, n], i) => (
                                            <tr>
                                                <td>x {n}</td>
                                                <td>{t}</td>
                                            </tr>
                                        )}
                                    </For>
                                </table>

                                <Show when={clusterReport()?.diagnostics}>
                                    {(d) => (
                                        <table>
                                            <caption>
                                                Cluster Diagnostics
                                            </caption>
                                            <tbody>
                                                <tr>
                                                    <th>Events</th>
                                                    <td>{d().clusterStats.totalEvents}</td>
                                                </tr>
                                                <tr>
                                                    <th>Multi-cluster events</th>
                                                    <td> {d().clusterStats.multiClusterEvents}</td>
                                                </tr>
                                                <tr>
                                                    <th>Multi-cluster rate</th>
                                                    <td>{d().clusterStats.multiClusterRate.toFixed(4)}</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                    )}
                                </Show>
                            </div>

                            <div class="s10">
                                <div class="large-height scroll surface">

                                    <table class="stripes no-border scroll max">
                                        <caption>
                                            Top Documents
                                        </caption>
                                        <thead class="fixed">
                                            <tr>
                                                <td></td>
                                                <td>Count</td>
                                                <td>Doc ID</td>
                                                <td>Author</td>
                                                <td>Year</td>
                                                <td>Title</td>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <For each={c().topDocs}>
                                                {([doc_id], i) => (
                                                    <DocRow
                                                        rank={i() + 1}
                                                        doc_id={doc_id}
                                                        count={clusterExemplarsByDoc()?.[doc_id].length}
                                                        author={clusterReport()?.docMeta[doc_id]?.author ?? null}
                                                        pub_year={clusterReport()?.docMeta[doc_id]?.pub_year ?? null}
                                                        title={clusterReport()?.docMeta[doc_id]?.title ?? null}
                                                        exemplars={
                                                            showExemplars() ? clusterExemplarsByDoc()?.[doc_id] ?? [] : []
                                                        }
                                                    />
                                                )}
                                            </For>
                                        </tbody>
                                    </table>

                                </div>
                            </div>
                        </div>
                    )}
                </Show>
            </Show>
        </article >
    );
}