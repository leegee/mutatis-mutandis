import { createSignal, createResource, Show, Switch, Match } from "solid-js";

import type { PointData, ViewBounds } from "./types";
import type { Id } from "./SelectionPlugin/types";

import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets, loadBfsDataset } from "./loadScatterDatasets";
import { controls, type ColorScatterByType, type ScatterPlotLayerType } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import SidebarMultiple from "../SidebarMultiple";
import GlobalMessageDisplay from "../GlobalMessageDisplay";
import TextWindow from "../TextWindow";
import { labelState } from "../../state/labels.store";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_label"];

export default function ConceptClusterPlot() {
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
        () => controls.scatterPlotLayerMode === "neighbours" ? { ...sharedKey(), dataType: "concept_neighbours" } : null,
        loadDatasets
    );

    const [clusterDatasets] = createResource(
        () => controls.scatterPlotLayerMode === "clusters" ? { ...sharedKey(), dataType: "concept_clusters" } : null,
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
        if (controls.scatterPlotLayerMode === "clusters") {
            return (clusterDatasets() ?? []).map(d => ({ ...d, origin: "clusters" }));
        }
        const concept = (conceptDatasets() ?? []).map(d => ({ ...d, origin: "concept" }));
        const neighbours = controls.scatterPlotLayerMode === "neighbours"
            ? (neighbourDatasets() ?? []).map(d => ({ ...d, origin: "neighbours" }))
            : [];
        return [...concept, ...neighbours];
    };

    // const loading = () => conceptDatasets.loading || bfs.loading || clusterDatasets.loading;
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
                                    controlsActions.setScatterplotLayerMode(e.currentTarget.value as "concept" | "neighbours" | "clusters");
                                    if (e.currentTarget.value === 'clusters') {
                                        controlsActions.setColorBy("cluster_label")
                                    }
                                }}>
                                <option value="neighbours">All</option>
                                <option value="concept">Concept</option>
                                <option value="clusters">Clusters</option>
                            </select>
                            <div class="tooltip bottom">
                                Show concepts with neighbours (All), concepts without neighbours, or clusters.
                            </div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={controls.colorScatterBy}
                                onChange={e => controlsActions.setColorBy(e.currentTarget.value as ColorScatterByType)}
                            >
                                {COLOR_FIELDS.filter(clrMode => !(controls.scatterPlotLayerMode !== 'clusters' && clrMode.includes('cluster')))
                                    .map(v => (
                                        <option value={v}>{
                                            v.replace('_', ' ').replace(/^(.)/, _ => _.toLocaleUpperCase())
                                        }</option>)
                                    )}
                            </select>
                            <div class="tooltip bottom">Choose the domain upon which to base the point colours </div>
                        </div>

                        <div class="field border middle-align">
                            <select class="small-padding" value={controls.projectionMode}
                                onChange={e => controlsActions.setProjection(e.currentTarget.value as "local" | "global")}>
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
                                projectionMode={controls.projectionMode}
                                datasets={activeDatasets()}
                                bfsDataset={bfs()}
                                bfsOpacity={controls.bfsOpacity}
                                neighbourOpacity={controls.neighbourOpacity}
                                colorBy={controls.colorScatterBy}
                                colorByFields={COLOR_FIELDS}
                                onPointHover={(pt, xy) =>
                                    setHovered(pt && xy ? { point: pt, x: xy[0], y: xy[1] } : null)
                                }
                                onBoundsChange={handleBoundsChange}
                                selected={controls.selectedEventIds}
                                onSelectionChange={handleSelectionChange}
                            />
                        </div>

                        <SidebarMultiple onClose={() => {
                            controlsActions.setSelectedEventIds(null);
                            // setParamsTokenIdx(null);
                        }} />
                    </div>
                </Show>
            </Show>

            {/* Hover Tooltip */}
            <Show when={hovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate no-padding"
                        style={{
                            position: "fixed",
                            left: `${ h().x + 100 }px`,
                            top: `${ h().y + 70 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "white-space": "nowrap",
                            "width": "20em",
                        }}>

                        <header class="bottom-margin">
                            <h2 class="medium-padding fill max">{h().point.token}</h2>

                            <Show when={h().point.depth
                                || (
                                    controls.scatterPlotLayerMode === 'clusters'
                                    && h().point.concept)
                            }>
                                <div class="medium-opacity small-text no-space small small-margin tiny-padding">
                                    <span class="max no-space small small-margin no-padding">
                                        <Switch>
                                            <Match when={controls.scatterPlotLayerMode === 'clusters'}>
                                                {h().point.pub_year}
                                                <Show when={h().point.concept}>
                                                    {h().point.concept}
                                                </Show>
                                            </Match>

                                            <Match when={controls.scatterPlotLayerMode !== 'clusters'}>
                                                <span class="bold">{h().point.pub_year} </span>
                                                <Show when={h().point.depth}>
                                                    {h().point.concept}
                                                    <sup class="medium-text"> {h().point.depth}</sup>
                                                </Show>
                                            </Match>
                                        </Switch>
                                    </span>

                                    <span>
                                        {h().point.cluster_label && ` Cluster ${ h().point.cluster_label }`}
                                    </span>
                                </div>
                            </Show>
                        </header>

                        <div class="left-padding right-padding">
                            <span class="medium-opacity">
                                Doc: {h().point.doc_id} T {h().point.token_idx}
                                <br />
                                Win: {h().point.window_id} T {h().point.window_token_pos}
                            </span>
                        </div>
                        <footer class="row border padding" style="bottom-padding: 1em; top-padding: 1em">
                            <TextWindow eventid={h().point.event_id} style="font-size: 8pt; line-height: 1.6;" />
                        </footer>
                    </aside>
                )}
            </Show>
        </>
    );
}
