// Note clustering is very WIP - clustering is by full corpus span, so centroids are global to that regardless of oyear filter
import { createSignal, createResource, Show, Switch, Match } from "solid-js";

import type { BfsDataset, PointData, ViewBounds } from "./types";

import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets, loadBfsDataset } from "./loadScatterDatasets.sqlite";
import { controls, type ColorScatterByType, type ProjectionModeType, } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import SidebarMultiple from "../SidebarMultiple";
import GlobalMessageDisplay from "../GlobalMessageDisplay";
import ClusterTooltip from "./ClusterTooltip";
import ConceptTooltip from "./ConceptTooltip";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_id"];

export default function ConceptClusterPlot() {
    const [pointHovered, setPointHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

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
        () => controls.showNeighbours ? { ...sharedKey(), dataType: "concept_neighbours" } : null,
        loadDatasets
    );

    const [clusterDatasets] = createResource(
        () => controls.showClusterCentroids ? { ...sharedKey(), dataType: "concept_clusters" } : null,
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

    // Concept is always shown; neighbours and clusters are layered on top when toggled.
    // BFS is always passed separately to Plot and rendered beneath.
    const activeDatasets = () => {
        const concept = (conceptDatasets() ?? [])
            .map(d => ({ ...d, origin: "concept" }));

        const neighbours = controls.showNeighbours
            ? (neighbourDatasets() ?? [])
                .map(d => ({ ...d, origin: "neighbours" }))
            : [];

        const clusters = controls.showClusterCentroids
            ? (clusterDatasets() ?? [])
                .map(d => ({ ...d, origin: "concept_clusters" }))
            : [];

        return [
            ...concept,
            ...neighbours,
            ...clusters,
        ];
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
                <progress class="absolute bottom" style="z-index:1000" />
            </Show>

            <Show when={conceptDatasets()}>
                <Show when={!error()} fallback={
                    <GlobalMessageDisplay
                        title="Failed to load plot data"
                        errorMessage={conceptDatasets.error?.message || bfs.error?.message}
                        retry={refetch}
                    />
                }>
                    <ControlsHeader multiConcept authorMatch>
                        <div class="field  border middle-align">
                            <label class="switch icon">
                                <input type="checkbox"
                                    checked={controls.showNeighbours}
                                    onInput={e => controlsActions.setShowNeighbours(e.currentTarget.checked)}
                                />
                                <span><i>tenancy</i></span>
                            </label>
                            <div class="tooltip bottom"> Show neighbours </div>
                        </div>

                        <div class="field  border middle-align">
                            <label class="switch icon">
                                <input type="checkbox"
                                    checked={controls.showClusterCentroids}
                                    onInput={e => controlsActions.setShowClusterCentroids(e.currentTarget.checked)}
                                />
                                <span><i>hive</i></span>
                            </label>
                            <div class="tooltip bottom"> Show cluster info </div>
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
                                                    disabled={!controls.showNeighbours}
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
                                showNeighbours={controls.showNeighbours}
                                showClusterCentroids={
                                    controls.projectionMode === "global" &&
                                    controls.showClusterCentroids
                                }
                                neighbourOpacity={controls.neighbourOpacity}
                                onBoundsChange={handleBoundsChange}
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

            {/* Hover Tooltip */}
            <Show when={pointHovered()}>
                {(hoveredPoint) => (
                    <aside class="surface-container-highest border large-elevate no-padding"
                        style={{
                            position: "fixed",
                            left: `${ hoveredPoint().x + 100 }px`,
                            top: `${ hoveredPoint().y + 70 }px`,
                            "z-index": 20,
                            "pointer-events": "none",
                            "width": "20em",
                        }}>

                        <Switch>
                            <Match when={hoveredPoint().point.origin === "concept_clusters"}>
                                <ClusterTooltip point={hoveredPoint().point} />
                            </Match>

                            <Match when={hoveredPoint().point.origin !== "concept_clusters"}>
                                <ConceptTooltip point={hoveredPoint().point} />
                            </Match>
                        </Switch>

                    </aside>
                )}
            </Show >
        </>
    );
}
