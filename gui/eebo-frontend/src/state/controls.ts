import { createStore } from "solid-js/store";
import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";

export type ViewMode = "aggregated" | "events";
export type YearMode = "single" | "range";

export type ControlsState = {
  concept: string;
  viewMode: ViewMode;
  maxHubs: number;
  topN: number;
  minSimilarity: number;
  hubSpread: number;
  selectedNode: string | null;
  yearMode: YearMode;
  fromYear: number;
  toYear: number;
};

export const [controls, setControls] = createStore<ControlsState>({
  concept: 'LIBERTY',
  viewMode: "aggregated",
  maxHubs: 50,
  topN: 5,
  minSimilarity: 0.5,
  hubSpread: 1,
  selectedNode: null,
  yearMode: "single",
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_END_YEAR,
});


export function setConcept(c: string) {
  setControls({
    concept: c,
    selectedNode: null,
  });
}

export function setYearMode(mode: YearMode, bounds: [number, number]) {
  const [min, max] = bounds;

  setControls({
    yearMode: mode,
    fromYear: mode === "single" ? Math.floor((min + max) / 2) : min,
    toYear: mode === "single" ? Math.floor((min + max) / 2) : max,
  });
}
