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

export type Neighbor = { token: string; similarity: number; count: number };
export type Slice = { year: number; drift: number; js_divergence: number; top_neighbors: Neighbor[] };
export type TokenData = { slices: Slice[]; phase_transitions?: any };
export type Dataset = { [token: string]: TokenData };

export type Selection = { token: string | null; year: number | null; color: string };

export type DriftChartProps = {
    data: Dataset | undefined;
    hovered: () => Selection | null;
    setHovered: (s: Selection | null) => void;
    selected: () => Selection;
    setSelected: (s: Selection) => void;
};

export type NeighborGraphProps = {
    token: string | null;
    neighbors: Neighbor[];
    drift: number;
    color: string;
};
