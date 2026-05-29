// src/state/controls.ts
import { createStore } from "solid-js/store";
import { CORPUS_START_YEAR } from "../corpus_config";

type ViewMode = "aggregated" | "events";
type YearMode = "single" | "range";

export const [controls, setControls] = createStore({
  concept: 'LIBERTY',
  viewMode: "aggregated" as ViewMode,

  maxHubs: 50,
  topN: 5,
  minSimilarity: 0.5,
  hubSpread: 1,

  selectedNode: null as string | null,

  yearMode: "single" as YearMode,
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_START_YEAR,
});

// Optional: export types too
export type { ViewMode, YearMode };
