import { createSignal, createResource, Show, Switch, Match, createEffect, onMount } from "solid-js";

import type { BfsDataset, LabelPoint, PointData, ViewBounds } from "./types";

import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets, loadBfsDataset } from "./loadScatterDatasets.sqlite";
import { controls, type ColorScatterByType, type ProjectionModeType, type ScatterPlotLayerType } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import SidebarMultiple from "../SidebarMultiple";
import GlobalMessageDisplay from "../GlobalMessageDisplay";
import TextWindow from "../TextWindow";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_label"];

console.log("[scatterplot] loaded");

export default function ConceptClusterPlot() {
    const [pointHovered, setPointHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);
    const [labelHovered, setLabelHovered] = createSignal<{ label: LabelPoint; x: number; y: number } | null>(null);

    const sharedKey = () => ({
        concepts: controls.conceptSelection,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        yearMode: controls.yearMode,
        authorMatch: controls.authorMatch,
    });

    const [conceptDatasets, { refetch }] = createResource(
        () => ({ ...sharedKey(), dataType: "concept" }),
        loadDatasets
    );

    const [neighbourDatasets] = createResource(
        () => controls.scatterPlotLayerMode === "neighbours" ? { ...sharedKey(), dataType: "concept_neighbours" } : null,
        loadDatasets
    );

    const [clusterDatasets] = createResource(
        () => controls.scatterPlotLayerMode === "concept_clusters" ? { ...sharedKey(), dataType: "concept_clusters" } : null,
        loadDatasets
    );

    const [bfs] = createResource(
        () => false // controls.showBfsGlobal
            ? {
                fromYear: controls.fromYear,
                toYear: controls.toYear,
                yearMode: controls.yearMode,
            }
            : null,
        loadBfsDataset
    );

    // Concept is always shown; neighbours are layered on top when toggled.
    // BFS is always passed separately to Plot and rendered beneath.
    const activeDatasets = () => {
        if (controls.scatterPlotLayerMode === "concept_clusters") {
            return (clusterDatasets() ?? []).map(d => ({ ...d, origin: "concept_clusters" }));
        }
        const concept = (conceptDatasets() ?? []).map(d => ({ ...d, origin: "concept" }));
        const neighbours = controls.scatterPlotLayerMode === "neighbours"
            ? (neighbourDatasets() ?? []).map(d => ({ ...d, origin: "neighbours" }))
            : [];
        return [...concept, ...neighbours];
    };

    const loading = () => conceptDatasets.loading || bfs.loading || clusterDatasets.loading;
    const error = () => conceptDatasets.error || bfs.error || clusterDatasets.error;

    function handleSelectionChange(points: PointData[] | null) {
        console.debug("[ScattPlot] handleSelectionChange event:", points);
        controlsActions.setSelectedPoints(points || []);
    }

    function handleBoundsChange(_bounds: ViewBounds) {
        // handle if needed
    }

    return (
        <>
            <Show when={loading()}>
                <progress class="absolute top" />
            </Show>
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
                            <select class="small-padding" value={controls.scatterPlotLayerMode}
                                onChange={e => {
                                    controlsActions.setScatterplotLayerMode(e.currentTarget.value as ScatterPlotLayerType);
                                    if (e.currentTarget.value === 'concept_clusters') {
                                        controlsActions.setColorBy("cluster_label")
                                    }
                                }}>
                                <option value="neighbours">All</option>
                                <option value="concept">Concept</option>
                                <option value="concept_clusters">Clusters</option>
                            </select>
                            <div class="tooltip bottom">
                                Show concepts with neighbours (All), concepts without neighbours, or clusters.
                            </div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={controls.colorScatterBy}
                                onChange={e => controlsActions.setColorBy(e.currentTarget.value as ColorScatterByType)}
                            >
                                {COLOR_FIELDS.map(v => (
                                    <option value={v}>{
                                        v.replace('_', ' ').replace(/^(.)/, _ => _.toLocaleUpperCase())
                                    }</option>)
                                )}
                            </select>
                            <div class="tooltip bottom">Choose the domain upon which to base the point colours </div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={controls.projectionMode}
                                onChange={e => controlsActions.setProjection(e.currentTarget.value as ProjectionModeType)}>
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
                                                <input type='range' min={0} max={4} step={0.5}
                                                    value={controls.bfsOpacity}
                                                    onInput={(e) => controlsActions.setBfsOpacity(e.currentTarget.value)}
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
                                                    disabled={controls.scatterPlotLayerMode !== "neighbours"}
                                                    value={controls.neighbourOpacity}
                                                    onInput={(e) => controlsActions.setNeighbourOpacity(Number(e.currentTarget.value))}
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
                                bfsDataset={bfs() as BfsDataset}
                                bfsOpacity={controls.bfsOpacity}
                                colorBy={controls.colorScatterBy}
                                colorByFields={COLOR_FIELDS}
                                datasets={activeDatasets()}
                                neighbourOpacity={controls.neighbourOpacity}
                                onBoundsChange={handleBoundsChange}
                                onLabelHover={(label, xy) => setLabelHovered(label && xy ? { label: label, x: xy[0], y: xy[1] } : null)}
                                onPointHover={(point, xy) => setPointHovered(point && xy ? { point: point, x: xy[0], y: xy[1] } : null)}
                                onSelectionChange={handleSelectionChange}
                                projectionMode={controls.projectionMode}
                                selectedEventIds={controls.selectedEventIds}
                            />
                        </div>

                        <SidebarMultiple onClose={() => {
                            controlsActions.setSelectedEventIds(null);
                        }} />
                    </div>
                </Show>
            </Show>

            {/* Label-hover tooltip */}
            <Show when={labelHovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate padding"
                        style={{
                            position: "fixed",
                            left: `${ h().x + 100 }px`,
                            top: `${ h().y + 70 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "width": "20em",
                        }}>
                        {JSON.stringify(h().label.description)}
                    </aside>
                )}
            </Show>

            {/* Piont-hover Tooltip */}
            <Show when={pointHovered()}>
                {(hoveredPoint) => (
                    <aside class="surface-container-highest border large-elevate no-padding"
                        style={{
                            position: "fixed",
                            left: `${ hoveredPoint().x + 100 }px`,
                            top: `${ hoveredPoint().y + 70 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "white-space": "nowrap",
                            "width": "20em",
                        }}>

                        <Switch>
                            <Match when={controls.scatterPlotLayerMode === 'concept_clusters'}>
                                <div class="padding">
                                    <h2>{hoveredPoint().point.cluster_label}</h2>
                                    <p>{hoveredPoint().point.concept}</p>
                                </div>
                            </Match>

                            <Match when={controls.scatterPlotLayerMode !== 'concept_clusters'}>
                                <header class="bottom-margin fill">
                                    <h2 class="fill max"><q>{hoveredPoint().point.token}</q></h2>
                                    <Show when={hoveredPoint().point.depth
                                        || (
                                            controls.scatterPlotLayerMode === 'concept_clusters'
                                            && hoveredPoint().point.concept)
                                    }>
                                        <div class="medium-opacity small-text no-space small small-margin tiny-padding">
                                            <span class="max no-space small small-margin no-padding">
                                                <span class="bold">{hoveredPoint().point.pub_year} </span>
                                                <Show when={hoveredPoint().point.depth}>
                                                    {" "}{hoveredPoint().point.concept}
                                                    <sup class="medium-text"> {hoveredPoint().point.depth}</sup>
                                                </Show>
                                            </span>
                                        </div>
                                    </Show>
                                </header>

                                <div class="left-padding right-padding">
                                    <span class="medium-opacity">
                                        Doc: {hoveredPoint().point.doc_id} T {hoveredPoint().point.token_idx}
                                        <br />
                                        Win: {hoveredPoint().point.window_id} T {hoveredPoint().point.window_token_pos}
                                    </span>
                                </div>

                                <div class="left-padding right-padding">
                                    <span class="medium-opacity">
                                        Cluster {hoveredPoint().point.cluster_label || 'N/A'}
                                    </span>
                                </div>

                                <footer class="row padding fill" style="bottom-padding: 1em; top-padding: 1em">
                                    <TextWindow eventid={hoveredPoint().point.event_id} style="font-size: 12pt; line-height: 1.6;" />
                                </footer>
                            </Match>
                        </Switch>
                    </aside>
                )}
            </Show >
        </>
    );
}
