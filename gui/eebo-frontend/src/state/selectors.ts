// src/state/selectors.ts

import { tier2Data } from "./tier2data.store";
import { controls } from "./controls.store";
import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { filterByYearRange, scanYearRange } from "../lib/contextGraphUtils";
import type { ConceptData, ConceptEvent } from "../types/context-graph.types";

export function getYearBounds(): [number, number] {
    const conceptData = tier2Data[controls.concept];
    if (!conceptData) {
        return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    }
    return scanYearRange(conceptData);
}

export function yearBoundsFrom(conceptData: ConceptData): [number, number] {
    return scanYearRange(conceptData);
}

export function getYearFiltered() {
    const cd = tier2Data[controls.concept];
    if (!cd) return [];

    const bounds = getYearBounds();

    return filterEvents(
        cd.events,
        controls.fromYear,
        controls.toYear,
        bounds
    );
}

export const totalEventsForConcept = () => {
    const cd = tier2Data[controls.concept];
    return cd?.n_events ?? 0;
};


export function filterEvents(
    events: ConceptEvent[],
    from: number,
    to: number,
    bounds?: [number, number]
) {
    if (bounds) {
        const [min, max] = bounds;
        if (from <= min && to >= max) return events;
    }
    return filterByYearRange(events, from, to);
}

