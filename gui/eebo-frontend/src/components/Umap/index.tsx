import { createSignal, createResource, Show } from "solid-js";

import type { ConceptDataset, PointData, ViewBounds } from "./types";
import UmapPlot from "./UmapPlot";
import ControlsHeader from "../ControlsHeader";
import { loadDatasets } from "./loadDatasets";

const COLOR_FIELDS = ["doc_id", "token", "concept"];

export default function Umap() {
    const [projection, setProjection] = createSignal<"local" | "global">("global");
    const [colorBy, setColorBy] = createSignal("doc_id");
    const [hovered, setHovered] = createSignal<{ point: PointData; x: number; y: number } | null>(null);

    const [datasets] = createResource<ConceptDataset[]>(async () => {
        return loadDatasets();
    });

    function handleClick(point: PointData) {
        console.log("[Umap.index] clicked", point.token_idx, point.token, point.doc_id);
    }

    function handleBoundsChange(_bounds: ViewBounds) {
        // parent store would receive this — ignored here
        // console.log("[Umap.index] bounds changed", bounds);
    }

    return (
        <>
            {/* Map fills the screen */}
            <Show when={datasets()}>
                <ControlsHeader />
                <UmapPlot
                    datasets={datasets()!}
                    projection={projection()}
                    colorBy={colorBy()}
                    colorByFields={COLOR_FIELDS}
                    onPointHover={(pt, xy) =>
                        setHovered(pt && xy ? { point: pt, x: xy[0], y: xy[1] } : null)
                    }
                    onPointClick={handleClick}
                    onBoundsChange={handleBoundsChange}
                />
            </Show>

            {/* Minimal HUD — in the real system these controls live in the parent store UI */}
            <aside style={{
                position: "fixed", top: "1rem", right: "1rem", "z-index": 10,
                display: "flex", gap: "0.5rem", "flex-direction": "column",
            }}>
                <div class="field border middle-align">
                    <select class="small-padding"
                        value={projection()}
                        onChange={(e) => setProjection(e.currentTarget.value as "local" | "global")}
                    >
                        <option value="global">Global</option>
                        <option value="local">Local</option>
                    </select>
                </div>

                <div class="field border middle-align">
                    <select class="small-padding"
                        value={colorBy()}
                        onChange={(e) => setColorBy(e.currentTarget.value)}
                    >
                        {COLOR_FIELDS.map((f) => <option value={f}>{f}</option>)}
                    </select>
                </div>
            </aside>

            <Show when={hovered()}>
                {(h) => (
                    <aside class="surface-container-highest border large-elevate small-padding" style={{
                        position: "fixed",
                        left: `${ h().x + 14 }px`,
                        top: `${ h().y - 100
                            }px`,
                        "z-index": 20,
                        "pointer-events": "none",
                        "white-space": "nowrap",
                    }}>
                        <div class="bold">{h().point.token}</div>
                        <div class="small-text">{h().point.doc_id}</div>
                    </aside>
                )}
            </Show>
        </>
    );
}
