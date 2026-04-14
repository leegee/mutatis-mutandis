import { createStore } from "solid-js/store";
import { createResource } from "solid-js";
import type { Dataset, Selection } from "../types";
import { fetchTokenClusters } from "../services/tokenClustersService";

let cached: Dataset | null = null;

const WINDOW_OFFSET = 12;

export const OVERLAY_SIZE = {
    width: 300,
    height: 300
};

function clampOverlay(x: number, y: number) {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    let cx: number;
    let cy: number;

    if (x + OVERLAY_SIZE.width + WINDOW_OFFSET <= vw) {
        cx = x + WINDOW_OFFSET;
    } else {
        cx = x - OVERLAY_SIZE.width - WINDOW_OFFSET;
    }

    if (y + OVERLAY_SIZE.height + WINDOW_OFFSET <= vh) {
        cy = y + WINDOW_OFFSET;
    } else {
        cy = y - OVERLAY_SIZE.height - WINDOW_OFFSET;
    }

    cx = Math.max(0, Math.min(cx, vw - OVERLAY_SIZE.width));
    cy = Math.max(0, Math.min(cy, vh - OVERLAY_SIZE.height));

    return { x: cx, y: cy };
}

export const [data] = createResource<Dataset>(async () => {
    cached ??= await fetchTokenClusters("drift_neighbors_micro_senses_slices.json");
    return cached;
});

const [eeboStore, setEeboStore] = createStore({
    year: 1625,

    // NEW: global slice navigation index (authoritative temporal cursor)
    sliceIndex: 0,

    selected: {
        token: null,
        slice_start: null,
        slice_end: null,
        color: null,
    } as Selection,
});

const setNullSelected = eeboStore.selected = {
    token: null,
    slice_start: null,
    slice_end: null,
    color: null,
} as Selection;


export { eeboStore, setEeboStore, setNullSelected };