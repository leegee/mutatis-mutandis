import { createStore } from "solid-js/store";

type ViewMode = "aggregated" | "events";

export const [controls, setControls] = createStore({
  concept: 'LIBERTY',
  viewMode: "aggregated" as ViewMode,

  maxHubs: 50,
  topN: 5,
  minSimilarity: 0.5,

  hubSpread: 1,

  selectedNode: null as string | null,

  fromYear: -1,
  toYear: -1,
  yearMode: "single" as "single" | "range",
});
