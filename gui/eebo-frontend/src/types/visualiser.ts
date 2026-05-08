export type DocWeights = Record<string, number>;

export type Tier3Node = {
    id: string;
    slice: string;
    cluster: number;
    size: number;
    centroid: number[] | null;
    vector_ids: number[];
    doc_weights: DocWeights;
    x?: number;
    y?: number;
};

export type Tier3Link = {
    source: string;
    target: string;
    similarity: number;
    weight: number;
    from_slice: string;
    to_slice: string;
};

export type Tier3TokenGraph = {
    token: string;
    nodes: Tier3Node[];
    links: Tier3Link[];
};

export type Tier3GraphData = Record<string, Tier3TokenGraph>;
