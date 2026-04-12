import { createStore } from "solid-js/store";
import { createResource } from "solid-js";
import type { Dataset, Selection } from "../types";
import { fetchTokenClusters } from "../services/tokenClustersService";

let cached: Dataset | null = null;

const WINDOW_OFFSET = 12;
export const OVERLAY_SIZE = {
    width: 750,
    height: 750
};


function clampOverlay(x: number, y: number) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let cx: number;
    let cy: number;

    // --- horizontal
    if (x + OVERLAY_SIZE.width + WINDOW_OFFSET <= vw) {
        // place to the right of cursor
        cx = x + WINDOW_OFFSET;
    } else {
        // place to the left
        cx = x - OVERLAY_SIZE.width - WINDOW_OFFSET;
    }

    // --- vertical
    if (y + OVERLAY_SIZE.height + WINDOW_OFFSET <= vh) {
        // place below cursor
        cy = y + WINDOW_OFFSET;
    } else {
        // place above cursor
        cy = y - OVERLAY_SIZE.height - WINDOW_OFFSET;
    }

    // --- final clamp safety (important for extreme edges)
    cx = Math.max(0, Math.min(cx, vw - OVERLAY_SIZE.width));
    cy = Math.max(0, Math.min(cy, vh - OVERLAY_SIZE.height));

    console.log(x, y, '→', cx, cy);

    return { x: cx, y: cy };
}


export function closeOverlay() {
    const prev = { ...eeboStore._overlay };
    setEeboStore("_overlay", {
        ...prev,
        open: false
    });
}


export function openOverlay(x: number, y: number) {
    const pos = clampOverlay(x, y);
    setEeboStore("_overlay", {
        x: pos.x,
        y: pos.y,
        open: true
    });
}


export function toggleOverlay(x: number, y: number) {
    const pos = clampOverlay(x, y);
    setEeboStore("_overlay", (prev) => {
        const isSame =
            prev.open &&
            Math.abs(prev.x - pos.x) < 1 &&
            Math.abs(prev.y - pos.y) < 1;

        return {
            x: pos.x,
            y: pos.y,
            open: !isSame
        };
    });
}


export const [data] = createResource<Dataset>(async () => {
    cached ??= await fetchTokenClusters("drift_neighbors_micro_senses_slices.json")
    console.log(`[Eebo.store] [Resource] data rv`, cached)
    return cached;
});

const [eeboStore, setEeboStore] = createStore({
    year: 1625,
    selected: {
        token: null,
        slice_start: null,
        slice_end: null,
        color: null,
    } as Selection,
    _overlay: {
        open: false,
        x: 0,
        y: 0
    }
});

export { eeboStore, setEeboStore };
