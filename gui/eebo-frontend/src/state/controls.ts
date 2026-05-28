import { createStore } from "solid-js/store";
import { CORPUS_START_YEAR } from "../corpus_config";

type ViewMode = "aggregated" | "events";

export const [controls, setControls] = createStore({
  concept: 'LIBERTY',
  viewMode: "aggregated" as ViewMode,

  maxHubs: 50,
  topN: 5,
  minSimilarity: 0.5,

  hubSpread: 1,

  selectedNode: null as string | null,

  yearMode: "single" as "single" | "range",
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_START_YEAR,
});
