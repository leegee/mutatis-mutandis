export type Neighbour = {
    neighbour_event_id: number;
    token: string;
    doc_id: string;
    pub_year: number;
    token_idx: number;
    score: number;
    depth: number;
};

export type EventSample = {
    event_id: string;
    doc_id: string;
    token_idx: number;
    token: string;
    pub_year: number;
    neighbours: Neighbour[];
};

export type LineageNode = {
    id: string;
    year: number;
    cluster: number;
    size: number;
    lineage?: number;
    merged_from?: number[];
    persistence_score?: number;
    lineage_stable?: boolean;
    event_sample?: EventSample[];
    local?: {
        x: number;
        y: number;
    };

    global?: {
        x: number;
        y: number;
    };
};


export type LineageLink = {
    source: string;
    target: string;
    similarity: number;
    confidence: number;
    type: string;
};

export type LineageData = {
    concept: string;
    nodes: LineageNode[];
    links: LineageLink[];
};

export type TooltipState = {
    node: LineageNode;
    x: number;
    y: number;
};


export type ViewportRatio = {
    // Both 0..1, expressed as a fraction of the detail view's full
    // scrollable width -- not pixels, so they stay meaningful regardless
    // of container size or how many years are in the timeline.
    startRatio: number;
    endRatio: number;
};

export type ScrollState = {
    scrollLeft: number;
    scrollWidth: number;
    clientWidth: number;
};