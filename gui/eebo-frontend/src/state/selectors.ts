// src/state/selectors.ts

import { tier2Data } from "./tier2data.store";
import { controls } from "./controls.store";
import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { scanYearRange } from "../lib/contextGraphUtils";

export function getYearBounds(): [number, number] {
    const conceptData = tier2Data[controls.concept];

    if (!conceptData) {
        return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    }

    return scanYearRange(conceptData);
}