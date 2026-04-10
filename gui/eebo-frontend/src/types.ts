export interface MyDocument {
    _id: string;
    _source: {
        title: string;
        author: string;
        year: number;
        place: string;
        publisher: string;
        text: string;
    };
}

export interface Hit {
    _id: string;
    _source: {
        title: string;
        author: string;
        year: number;
        place: string;
    };
}





// todo move to a new file

export type SlicePoint = {
    slice_start: number;
    slice_end: number;
    drift: number;
    transitions?: number[];
};

export type Neighbor = {
    token: string;
    similarity: number;
    count: number;
};

export type Slice = {
    slice_start: number;
    slice_end: number;

    n_clusters: number;
    cluster_sizes: number[];

    entropy: number;

    top_neighbors: Neighbor[];

    count: number;
    top_docs: [string, number][];

    drift: number;
    births: number;
    deaths: number;
    js_divergence: number;
};

export type TokenData = {
    slices: Slice[];
    phase_transitions?: any;
};

export type Dataset = {
    [token: string]: TokenData;
};

export type SliceHistoryPoint = {
    t: number;
    drift: number;
    d1: number;
    d2: number;
};

export type SliceView = {
    token: string;
    slice_start: number;
    slice_end: number;

    neighbors: Neighbor[];
    drift: number;
    normalizedDrift: number;

    rank: Map<string, number>;

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
}
