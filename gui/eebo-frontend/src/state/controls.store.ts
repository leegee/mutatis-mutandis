// src/state/controls.store.ts

import { createStore } from "solid-js/store";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

export type ViewMode = "aggregated" | "events";
export type YearMode = "single" | "range";

export const MAX_TOP_N = 1000;


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
  showEventLabels: boolean;
};

export const [controls, setControls] = createStore<ControlsState>({
  concept: "LIBERTY",
  viewMode: "events",
  maxHubs: 50,
  topN: MAX_TOP_N,
  minSimilarity: 0.5,
  hubSpread: 1,
  selectedNode: null,
  yearMode: "range",
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_END_YEAR,
  showEventLabels: false,
});
