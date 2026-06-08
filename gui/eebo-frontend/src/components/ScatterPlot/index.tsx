import { createSignal, createResource, Show } from "solid-js";

import type { PointData, ViewBounds } from "./types";
import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets, loadBfsDataset } from "./loadScatterDatasets";
import { controls } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import Sidebar from "./Sidebar";
import GlobalMessageDisplay from "../GlobalMessageDisplay";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_label"];
const DATA_TYPES = ['concept_neighbours', 'concept', 'concept_clusters'];

export default function ConceptClusterPlot() {
    const [projection, setProjection] = createSignal<"local" | "global">("global");
    const [dataType, setDataType] = createSignal("concept_clusters");
    const [colorBy, setColorBy] = createSignal("cluster_label");
    const [hovered, setHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

    const [datasets] = createResource(
        () => ({
            concepts: controls.conceptSelection,
            fromYear: controls.fromYear,
            toYear: controls.toYear,
            yearMode: controls.yearMode,
            dataType: dataType()
        }),
        loadDatasets
    );

    const loading = () => datasets.loading || bfs.loading;
    const error = () => datasets.error || bfs.error;

    const [bfs] = createResource(
        () => ({
            fromYear: controls.fromYear,
            toYear: controls.toYear,
            yearMode: controls.yearMode,
        }),
        loadBfsDataset
    );

    function handleClick(point: PointData) {
        console.log("[ConceptClusterPlot] clicked event:", point.event_id);
        controlsActions.setSelectedEventId(point.event_id);
    }

    function handleBoundsChange(_bounds: ViewBounds) {
        // handle if needed
    }

    return (
        <>
            <Show when={datasets()}>
                <Show when={!error()} fallback={
                    <GlobalMessageDisplay
                        title="Failed to load plot data"
                        errorMessage={datasets.error?.message || bfs.error?.message}
                    />
                }>
                    <ControlsHeader multiConcept={true} noTopN={true}>
                        <div class="field border middle-align">
                            <select class="small-padding" value={dataType()}
                                onChange={e => setDataType(e.currentTarget.value)}>
                                {DATA_TYPES.map(t => <option value={t}>{t}</option>)}
                            </select>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={projection()}
                                onChange={e => setProjection(e.currentTarget.value as "local" | "global")}>
                                <option value="global">Global</option>
                                <option value="local">Local</option>
                            </select>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={colorBy()}
                                onChange={e => setColorBy(e.currentTarget.value)}>
                                {COLOR_FIELDS.map(f => <option value={f}>{f}</option>)}
                            </select>
                        </div>
                    </ControlsHeader>

                    <div id="graph_sidebar_row">
                        <div id="under_sidebar" class="max">
                            <Plot
                                datasets={datasets()!}
                                bfsDataset={bfs()}
                                projection={projection()}
                                colorBy={colorBy()}
                                colorByFields={COLOR_FIELDS}
                                onPointHover={(pt, xy) =>
                                    setHovered(pt && xy ? { point: pt, x: xy[0], y: xy[1] } : null)
                                }
                                onPointClick={handleClick}
                                onBoundsChange={handleBoundsChange}
                            />
                        </div>

                        <Sidebar />
                    </div>
                </Show>
            </Show>

            {/* Hover Tooltip */}
            <Show when={hovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate small-padding"
                        style={{
                            position: "fixed",
                            left: `${ h().x - 80 }px`,
                            top: `${ h().y - 80 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "white-space": "nowrap"
                        }}>
                        <div class="row">
                            <span class="bold max">{h().point.token}</span>
                            <span class="medium-opacity small-text padding-left">
                                {h().point.concept} • {h().point.cluster_label ? `Cluster ${ h().point.cluster_label }` : 'Noise'}
                            </span>
                        </div>
                        <div class="row small-text">Doc: {h().point.doc_id} • Year: {h().point.pub_year}</div>
                        <div class="row small-text">Win: {h().point.window_id} • Win token pos: {h().point.window_token_pos}</div>
                    </aside>
                )}
            </Show>
        </>
    );
}

