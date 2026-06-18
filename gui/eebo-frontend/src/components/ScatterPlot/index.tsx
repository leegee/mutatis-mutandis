import { createSignal, createResource, Show } from "solid-js";

import type { PointData, ViewBounds } from "./types";
import type { Id } from "./SelectionPlugin/types";

import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets, loadBfsDataset } from "./loadScatterDatasets";
import { controls } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import SidebarMultiple from "../SidebarMultiple";
import GlobalMessageDisplay from "../GlobalMessageDisplay";
import TextWindow from "../TextWindow";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_label"];

export default function ConceptClusterPlot() {
    const [projection, setProjection] = createSignal<"local" | "global">("global");
    const [layerMode, setLayerMode] = createSignal<"concept" | "neighbours" | "clusters">("concept");
    const [colorBy, setColorBy] = createSignal("pub_year");
    const [bfsOpacity, setBfsOpacity] = createSignal(3);
    const [neighbourOpacity, setNeighbourOpacity] = createSignal(200);
    const [hovered, setHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

    const sharedKey = () => ({
        concepts: controls.conceptSelection,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        yearMode: controls.yearMode,
    });

    const [conceptDatasets, { refetch }] = createResource(
        () => ({ ...sharedKey(), dataType: "concept" }),
        loadDatasets
    );

    const [neighbourDatasets] = createResource(
        () => layerMode() === "neighbours" ? { ...sharedKey(), dataType: "concept_neighbours" } : null,
        loadDatasets
    );

    const [clusterDatasets] = createResource(
        () => layerMode() === "clusters" ? { ...sharedKey(), dataType: "concept_clusters" } : null,
        loadDatasets
    );

    const [bfs] = createResource(
        () => ({
            fromYear: controls.fromYear,
            toYear: controls.toYear,
            yearMode: controls.yearMode,
        }),
        loadBfsDataset
    );

    // Concept is always shown; neighbours are layered on top when toggled.
    // BFS is always passed separately to Plot and rendered beneath.
    const activeDatasets = () => {
        if (layerMode() === "clusters") {
            return (clusterDatasets() ?? []).map(d => ({ ...d, origin: "clusters" }));
        }
        const concept = (conceptDatasets() ?? []).map(d => ({ ...d, origin: "concept" }));
        const neighbours = layerMode() === "neighbours"
            ? (neighbourDatasets() ?? []).map(d => ({ ...d, origin: "neighbours" }))
            : [];
        return [...concept, ...neighbours];
    };

    // const loading = () => conceptDatasets.loading || bfs.loading || clusterDatasets.loading;
    const error = () => conceptDatasets.error || bfs.error || clusterDatasets.error;

    function handleSelectionChange(event_ids: Id[] | null) {
        // console.debug("[ConceptClusterPlot] handleSelectionChange event:", event_ids ? event_ids.map(_ => _) : null);
        controlsActions.setSelectedEventIds(new Set(event_ids))
    }

    function handleBoundsChange(_bounds: ViewBounds) {
        // handle if needed
    }

    return (
        <>
            <Show when={conceptDatasets()}>
                <Show when={!error()} fallback={
                    <GlobalMessageDisplay
                        title="Failed to load plot data"
                        errorMessage={conceptDatasets.error?.message || bfs.error?.message}
                        retry={refetch}
                    />
                }>
                    <ControlsHeader multiConcept={true}>
                        <div class="field  border middle-align">
                            <select class="small-padding" value={layerMode()}
                                onChange={e => {
                                    setLayerMode(e.currentTarget.value as "concept" | "neighbours" | "clusters");
                                    if (e.currentTarget.value === 'clusters') {
                                        setColorBy("cluster_label")
                                    }
                                }}>
                                <option value="concept">Concept</option>
                                <option value="neighbours">+ Neighbours</option>
                                <option value="clusters">Clusters</option>
                            </select>
                            <div class="tooltip bottom">View mode</div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={colorBy()}
                                onChange={e => setColorBy(e.currentTarget.value)}>
                                {COLOR_FIELDS
                                    .filter(clrMode => !(clrMode.includes('cluster') && layerMode() !== 'clusters'))
                                    .map(
                                        v => <option value={v}>{
                                            v.replace('_', ' ').replace(/^(.)/, _ => _.toLocaleUpperCase())
                                        }</option>)}
                            </select>
                            <div class="tooltip bottom">Point colour mode</div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={projection()}
                                onChange={e => setProjection(e.currentTarget.value as "local" | "global")}>
                                <option value="global">Global</option>
                                <option value="local">Local</option>
                            </select>
                            <div class="tooltip bottom">Projection mode</div>
                        </div>

                        <div class="no-round bottom">
                            <button class="transparent circle">
                                <i>more_vert</i>
                            </button>
                            <menu class="no-round  bottom left no-wrap">

                                <li class="middle-align top-padding">
                                    <div class="field middle-align prefix suffix">
                                        <nav>
                                            <div class="slider medium responsive">
                                                <input type='range' min={0} max={25} step={1}
                                                    value={bfsOpacity()}
                                                    onInput={(e) => setBfsOpacity(Number(e.currentTarget.value))}
                                                />
                                                <span><i>brightness_6</i></span>
                                            </div>
                                        </nav>
                                        <output>BFS Background Opacity</output>
                                    </div>
                                </li>

                                <li class="middle-align top-padding">
                                    <div class="field middle-align prefix suffix">
                                        <nav>
                                            <div class="slider medium responsive">
                                                <input type='range' min={1} max={255} step={1}
                                                    disabled={layerMode() !== "neighbours"}
                                                    value={neighbourOpacity()}
                                                    onInput={(e) => setNeighbourOpacity(Number(e.currentTarget.value))}
                                                />
                                                <span><i>brightness_6</i></span>
                                            </div>
                                        </nav>
                                        <output>Neighbour Opacity</output>
                                    </div>
                                </li>

                            </menu>
                            <span class="tooltip right">More...</span>
                        </div>
                    </ControlsHeader>

                    <div id="graph_sidebar_row">
                        <div id="under_sidebar" class="max">
                            <Plot
                                projection={projection()}
                                datasets={activeDatasets()}
                                bfsDataset={bfs()}
                                bfsOpacity={bfsOpacity()}
                                neighbourOpacity={neighbourOpacity()}
                                colorBy={colorBy()}
                                colorByFields={COLOR_FIELDS}
                                onPointHover={(pt, xy) =>
                                    setHovered(pt && xy ? { point: pt, x: xy[0], y: xy[1] } : null)
                                }
                                onBoundsChange={handleBoundsChange}
                                selected={controls.selectedEventIds}
                                onSelectionChange={handleSelectionChange}
                            />
                        </div>

                        <SidebarMultiple />
                    </div>
                </Show>
            </Show>

            {/* Hover Tooltip */}
            <Show when={hovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate small-padding"
                        style={{
                            position: "fixed",
                            left: `${ h().x + 100 }px`,
                            top: `${ h().y + 70 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "white-space": "nowrap",
                            "width": "15em",
                            "max-width": "15em",
                        }}>
                        <div class="row">
                            <span class="bold max">{h().point.token}</span>
                            <span class="medium-opacity small-text padding-left">
                                {h().point.concept} • {h().point.cluster_label ? `Cluster ${ h().point.cluster_label }` : 'Noise'}
                            </span>
                        </div>
                        <div class="row">
                            <div class="row small-text">Doc: {h().point.doc_id} • Year: {h().point.pub_year}</div>
                        </div>
                        <div class="row">
                            <div class="row small-text">Win: {h().point.window_id} • Win token pos: {h().point.window_token_pos}</div>
                        </div>
                        <div class="row">
                            <TextWindow eventid={h().point.event_id} />
                        </div>
                    </aside>
                )}
            </Show>
        </>
    );
}
