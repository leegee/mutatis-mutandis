import { createResource, createSignal, Show } from "solid-js";
import { getUmapData } from "~/server/tier3/umap-query";
import { controlsActions } from "~/state/controls.actions";
import { controls } from "~/state/controls.store";
import ConceptTooltip from "./ConceptTooltip";
import Plot from "./Plot";
import type { UmapPoint } from "./types";

export default function ScatterPlot() {
    const [hoveredPoint, setHoveredPoint] = createSignal<{
        point: UmapPoint;
        screenXY: [number, number];
    } | null>(null);

    const [dataset] = createResource(
        () => [controls.conceptSelection[0], controls.yearMode, controls.fromYear, controls.toYear] as const,
        ([concept, yearMode, fromYear, toYear]) => getUmapData(concept, yearMode, fromYear, toYear),
    );

    return (
        <Show when={dataset()} fallback={<div>Loading…</div>}>
            {(data) => (
                <>
                    <Plot
                        dataset={data()}
                        plotPointScaleFactor={1}
                        selectedEventIds={controls.selectedEventIds}
                        onSelectionChange={controlsActions.setSelectedEventIds}
                        onPointHover={(point, screenXY) => {
                            if (point && "eventId" in point && screenXY) {
                                setHoveredPoint({
                                    point,
                                    screenXY,
                                });
                            } else {
                                setHoveredPoint(null);
                            }
                        }}
                    />

                    <Show when={hoveredPoint()}>
                        {(hover) => {
                            const value = hover();

                            return (
                                <div style={{
                                    position: "absolute",
                                    left: `${ value.screenXY[0] + 12 }px`,
                                    top: `${ value.screenXY[1] + 12 }px`,
                                    "pointer-events": "none",
                                    "z-index": 10,
                                }}
                                >
                                    <ConceptTooltip point={value.point} />
                                </div>
                            );
                        }}
                    </Show>

                </>
            )}
        </Show>
    );
}
