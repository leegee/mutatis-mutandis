import { createStore } from "solid-js/store";
import { createResource } from "solid-js";
import type { Dataset, Selection } from "../types";
import { fetchTokenClusters } from "../services/tokenClustersService";

let cached: Dataset | null = null;

export const OVERLAY_SIZE = {
    width: 300,
    height: 300
};


export const [data] = createResource<Dataset>(async () => {
    cached ??= await fetchTokenClusters("drift_neighbors_micro_senses_slices.json");
    return cached;
});

const [eeboStore, setEeboStore] = createStore({
    year: 1625,

    sliceIndex: 0,

    selected: {
        token: null,
        slice_start: null,
        slice_end: null,
        color: null,
    } as Selection,
});

const setNullSelected = () => eeboStore.selected = {
    token: null,
    slice_start: null,
    slice_end: null,
    color: null,
} as Selection;


export { eeboStore, setEeboStore, setNullSelected };