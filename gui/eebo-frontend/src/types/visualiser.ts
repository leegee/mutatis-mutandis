export type SlicePoint = Slice & {
    transitions?: number[];
};

export type NamedSlicePoint = SlicePoint & {
    term: string;
};


// 🔴 UPDATED: neighbors are now mass-based projections
export type Neighbor = {
    token: string;
    mass: number;
};


// 🔴 backend-aligned slice structure
export type Slice = {
    slice_start: number;
    slice_end: number;
    corpus_count: number;
    support_count: number;
    top_neighbors: Neighbor[];
    top_docs: [string, number][];
    drift: number;
};


// 🔴 CRITICAL FIX:
// your code accesses dataset[token][sliceKey]
// not dataset[token].slices[]
export type TokenData = {
    [sliceKey: string]: Slice;
};


// dataset is unchanged structurally
export type Dataset = {
    [token: string]: TokenData;
};


export type SliceHistoryPoint = {
    t: number;
    drift: number;
    d1: number;
    d2: number;
};


// 🔴 UPDATED: rank now carries mass too
export type SliceView = {
    token: string;
    slice_start: number;
    slice_end: number;

    neighbors: Neighbor[];
    drift: number;
    normalizedDrift: number;

    rank: Map<string, { rank: number; mass: number }>;

    history: SliceHistoryPoint[];
    transitions: number[];
};


export type Selection = {
    token: string | null;
    slice_start: number | null;
    slice_end: number | null;
    color: string | null;
};


export type EventNeighbourhoodOpen = {
    token: string;
    slice_start: number;
    slice_end: number;
    color: string;
    x: number;
    y: number;
};


// ⚠️ This looks like an older experimental type
// left intact but aligned to new Neighbor definition
export type SemanticSlice = {
    slice_start: number;
    slice_end: number;

    count: number;
    support_mass: number;
    entropy: number;
    cluster_dispersion: number;
    js_drift: number;

    top_neighbors: Neighbor[];
};
