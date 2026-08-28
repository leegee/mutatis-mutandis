// Note clustering is very WIP - clustering is by full corpus span, so centroids are global to that regardless of oyear filter

import { useSearchParams } from "@solidjs/router";
import { createEffect, createResource, createSignal, Match, Show, Switch } from "solid-js";
import { controlsActions } from "../../state/controls.actions";
import { selectIds } from "../../state/controls.selectors";
import { type ColorScatterByType, controls, } from "../../state/controls.store";
import ControlsHeader from "../ControlsHeader";
import GlobalMessageDisplay from "../GlobalMessageDisplay";
import SidebarMultiple from "../SidebarMultiple";
import ClusterTooltip from "./ClusterTooltip";
import ConceptTooltip from "./ConceptTooltip";
import { loadBfsDataset, loadDatasets } from "./loadScatterDatasets.sqlite";
import Plot from "./Plot";
import PlotSettings from "./PlotSettings";
import type { BfsDataset, PointData, ViewBounds } from "./types";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept", "cluster_id"];

export default function ConceptClusterPlot() {
    const [searchParams] = useSearchParams();
    const [pointHovered, setPointHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

    const sharedKey = () => ({
        concepts: controls.conceptSelection,
        fromYear: controls.fromYear,
        toYear: controls.toYear,
        yearMode: controls.yearMode,
        authorMatch: controls.authorMatch,
    });

    const [conceptDatasets, { refetch }] = createResource(() => ({ ...sharedKey(), dataType: "concept" }), loadDatasets);

    const [neighbourDatasets] = createResource(
        () => (controls.showNeighbours ? { ...sharedKey(), dataType: "concept_neighbours" } : null),
        loadDatasets,
    );

    const [clusterDatasets] = createResource(
        () => (controls.showClusterCentroids ? { ...sharedKey(), dataType: "concept_clusters" } : null),
        loadDatasets,
    );

    const [bfs] = createResource(
        () =>
            // biome-ignore lint/correctness/noConstantCondition: <wip>
            false // controls.showBfsGlobal // controls.showBfsGlobal
                ? {
                    fromYear: controls.fromYear,
                    toYear: controls.toYear,
                    yearMode: controls.yearMode,
                }
                : null,
        loadBfsDataset,
    );

    // Concept is always shown; neighbours and clusters are layered on top when toggled.
    // BFS is always passed separately to Plot and rendered beneath.
    const activeDatasets = () => {
        const concept = (conceptDatasets() ?? []).map((d) => ({ ...d, origin: "concept" }));

        const neighbours = controls.showNeighbours
            ? (neighbourDatasets() ?? []).map((d) => ({ ...d, origin: "neighbours" }))
            : [];

        const clusters = controls.showClusterCentroids
            ? (clusterDatasets() ?? []).map((d) => ({ ...d, origin: "concept_clusters" }))
            : [];

        return [...concept, ...neighbours, ...clusters];
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

    // Select events where doc_id == the doc_id passed in the URL query string
    createEffect(() => {
        const docId = searchParams.doc_id;
        if (!docId) {
            return;
        }

        const points = activeDatasets().flatMap((d) => d.points ?? []);
        if (!points.length) return;

        const ids = selectIds(
            points,
            (p) => p.doc_id === docId,
            (p) => p.event_id,
        );

        console.info(`[ScatterPlot] found doc_id ${ docId } in URL and matched ${ ids.size } points`);
        controlsActions.setSelectedEventIds(ids);
    });

    function hoverPosition(x: number, y: number) {
        const xo = x >= document.body.clientWidth / 2 ? 100 : -190;
        const yo = y >= document.body.clientHeight / 2 ? -350 : 50;
        return {
            left: `${ x + xo }px`,
            top: `${ y + yo }px`,
        };
    }

    return (
        <>
            <Show when={loading()}>
                <progress class="absolute bottom" style="z-index:1000" />
            </Show>

            <Show when={conceptDatasets()}>
                <Show
                    when={!error()}
                    fallback={
                        <GlobalMessageDisplay
                            title="Failed to load plot data"
                            errorMessage={conceptDatasets.error?.message || bfs.error?.message}
                            retry={refetch}
                        />
                    }
                >
                    <ControlsHeader multiConcept authorMatch>
                        <div class="field  border middle-align">
                            <label class="switch icon">
                                <input
                                    type="checkbox"
                                    checked={controls.showNeighbours}
                                    onInput={(e) => controlsActions.setShowNeighbours(e.currentTarget.checked)}
                                />
                                <span>
                                    <i>tenancy</i>
                                </span>
                            </label>
                            <div class="tooltip bottom"> Show neighbours </div>
                        </div>

                        <div class="field  border middle-align">
                            <label class="switch icon">
                                <input
                                    type="checkbox"
                                    checked={controls.showClusterCentroids}
                                    onInput={(e) => controlsActions.setShowClusterCentroids(e.currentTarget.checked)}
                                />
                                <span>
                                    <i>hive</i>
                                </span>
                            </label>
                            <div class="tooltip bottom"> Show cluster info </div>
                        </div>

                        <div class="field border middle-align">
                            <select
                                class="small-padding"
                                value={controls.colorScatterBy}
                                onChange={(e) => controlsActions.setColorBy(e.currentTarget.value as ColorScatterByType)}
                            >
                                {COLOR_FIELDS.map((v) => (
                                    <option value={v}>{v.replace("_", " ").replace(/^(.)/, (_) => _.toLocaleUpperCase())}</option>
                                ))}
                            </select>
                            <div class="tooltip bottom">Choose the domain upon which to base the point colours </div>
                        </div>

                        <div class="field border middle-align">
                            <select
                                class="small-padding"
                                value={'local' /*controls.projectionMode*/}
                            // onChange={(e) => controlsActions.setProjection(e.currentTarget.value as ProjectionModeType)}
                            >
                                <option value="global">Global</option>
                                <option value="local">Local</option>
                            </select>
                            <div class="tooltip bottom">Projection mode</div>
                        </div>

                        <div class="no-round bottom">
                            <button type="button" class="transparent circle">
                                <i>settings</i>
                            </button>
                            <menu class="no-round  bottom left no-wrap">
                                <PlotSettings />
                            </menu>
                            <span class="tooltip bottom">Plot Settings</span>
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
                                onPointHover={(point, xy) => setPointHovered(point && xy ? { point: point, x: xy[0], y: xy[1] } : null)}
                                onSelectionChange={handleSelectionChange}
                                plotPointScaleFactor={controls.plotPointScaleFactor}
                                projectionMode={controls.projectionMode}
                                selectedEventIds={controls.selectedEventIds}
                                showNeighbours={controls.showNeighbours}
                                showClusterCentroids={controls.projectionMode === "global" && controls.showClusterCentroids}
                            />
                        </div>

                        <SidebarMultiple
                            onClose={() => {
                                controlsActions.setSelectedEventIds(null);
                            }}
                        />
                    </div>
                </Show>
            </Show>

            {/* Hover Tooltip */}
            <Show when={pointHovered()}>
                {(hoveredPoint) => (
                    <aside
                        class="surface-container-highest border large-elevate no-padding"
                        style={{
                            position: "fixed",
                            ...hoverPosition(hoveredPoint().x, hoveredPoint().y),
                            "z-index": 20,
                            "pointer-events": "none",
                            width: "20em",
                        }}
                    >
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
            </Show>
        </>
    );
}
