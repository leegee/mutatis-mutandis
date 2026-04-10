import { createStore } from "solid-js/store";
import { createResource } from "solid-js";
import type { Dataset, Selection } from "../types";
import { fetchTokenClusters } from "../services/tokenClustersService";

let cached: Dataset | null = null;

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
    } as Selection
});

export { eeboStore, setEeboStore };
