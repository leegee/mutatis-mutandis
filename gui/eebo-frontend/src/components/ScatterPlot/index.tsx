import { createSignal, createResource, Show } from "solid-js";

import type { PointData, ViewBounds } from "./types";
import Plot from "./Plot";
import ControlsHeader from "../ControlsHeader";
import { loadBfsDataset, loadDatasets } from "./loadScatterDatasets";
import { controls } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import Sidebar from "./Sidebar";

const COLOR_FIELDS = ["doc_id", "pub_year", "concept"];
const DATA_TYPES = ['concept_neighbours/', 'concept'];

export default function ScatterPlot() {
    const [projection, setProjection] = createSignal<"local" | "global">("global");
    const [dataType, setDataType] = createSignal("concept");
    const [colorBy, setColorBy] = createSignal("doc_id");
    const [hovered, setHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

    const [datasets] = createResource(
        () => ({
            concepts: controls.conceptSelection,
            fromYear: controls.fromYear,
            toYear: controls.toYear,
            yearMode: controls.yearMode,
            dataType: dataType()
        }),
        async ({ concepts, fromYear, toYear, yearMode, dataType }) => {
            return loadDatasets({
                concepts,
                fromYear,
                toYear,
                yearMode,
                dataType
            });
        }
    );

    const [bfs] = createResource(
        () => ({
            fromYear: controls.fromYear,
            toYear: controls.toYear,
            yearMode: controls.yearMode,
        }),
        async (params) => {
            console.log("[ScatterPlot.index] loaded BFS")
            return loadBfsDataset(params);
        }
    );

    async function handleClick(point: PointData) {
        console.log("[ScatterPlot.index] clicked", point.event_id);
        controlsActions.setSelectedEventId(point.event_id)
    }

    function handleBoundsChange(_bounds: ViewBounds) {
        // parent store would receive this — ignored here
        // console.log("[ScatterPlot.index] bounds changed", bounds);
    }

    return (
        <>
            {/* Map fills the screen */}
            <Show when={datasets() && bfs()}>
                <ControlsHeader multiConcept={true} noTopN={true} >
                    <div class="field border middle-align">
                        <select class="small-padding"
                            value={dataType()}
                            onChange={(e) => setDataType(e.currentTarget.value)}
                        >
                            {DATA_TYPES.map((t) => <option value={t}>{t}</option>)}
                        </select>
                        <span class="tooltip bottom">Colour by</span>
                    </div>

                    <div class="field border middle-align">
                        <select class="small-padding"
                            value={projection()}
                            onChange={(e) => setProjection(e.currentTarget.value as "local" | "global")}
                        >
                            <option value="global">Global</option>
                            <option value="local">Local</option>
                        </select>
                        <span class="tooltip bottom">Projection space</span>
                    </div>

                    <div class="field border middle-align">
                        <select class="small-padding"
                            value={colorBy()}
                            onChange={(e) => setColorBy(e.currentTarget.value)}
                        >
                            {COLOR_FIELDS.map((f) => <option value={f}>{f}</option>)}
                        </select>
                        <span class="tooltip bottom">Colour by</span>
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

            <Show when={hovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate small-padding" style={{
                        position: "fixed",
                        left: `${ h().x - 72 }px`,
                        top: `${ h().y - 72 }px`,
                        "z-index": 20,
                        "pointer-events": "none",
                        "white-space": "nowrap",
                    }}>
                        <div class="row">
                            <span class="bold max">
                                {h().point.token}
                            </span>
                            <span class="medium-opacity small-text padding-left">{h().point.concept}</span>
                        </div>
                        <div class="row small-text">Doc: {h().point.doc_id} Year: {h().point.pub_year}</div>
                        <div class="row small-text small-opacity">Token Index: {h().point.token_idx}</div>
                    </aside>
                )}
            </Show>
        </>
    );
}
