// src/state/selectors.ts

import { tier2Data } from "./tier2data.store";
import { controls } from "./controls.store";
import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { filterByYearRange, scanYearRange } from "../lib/contextGraphUtils";

export function getYearBounds(): [number, number] {
    const conceptData = tier2Data[controls.concept];

    if (!conceptData) {
        return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    }

    return scanYearRange(conceptData);
}

export function getYearFiltered() {
    const cd = tier2Data[controls.concept];
    if (!cd) return [];

    const [min, max] = getYearBounds();
    const events = cd.events;

    if (controls.fromYear <= min && controls.toYear >= max) {
        return events;
    }

    return filterByYearRange(events, controls.fromYear, controls.toYear);
}
