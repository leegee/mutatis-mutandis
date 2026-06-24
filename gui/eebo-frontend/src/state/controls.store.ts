// src/state/controls.store.ts

import { createStore } from "solid-js/store";
import { makePersisted } from "@solid-primitives/storage";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import type { ViewMode, YearMode } from "../types";

export const MAX_TOP_N = 1000;

export type ProjectionModeType = "local" | "global"
export type ScatterPlotLayerType = "concept" | "neighbours" | "clusters"
export type ColorScatterByType = 'pub_year' | "cluster_label" | "doc_id"

export type ControlsState = {
  concept: string;
  conceptSelection: string[];
  scatterPlotLayerMode: ScatterPlotLayerType;
  colorScatterBy: ColorScatterByType;
  viewMode: ViewMode;
  maxHubs: number;
  topN: number;
  minSimilarity: number;
  hubSpread: number;
  selectedNode: string | null;
  selectedEventId: string | null;
  selectedEventIds: Set<string> | null;
  yearMode: YearMode;
  fromYear: number;
  toYear: number;
  showEventLabels: boolean;
  projectionMode: ProjectionModeType;
};

const initialControls: ControlsState = {
  concept: "LIBERTY",
  conceptSelection: ["LIBERTY"],
  viewMode: "events",
  maxHubs: 50,
  topN: MAX_TOP_N,
  minSimilarity: 0.5,
  hubSpread: 1,
  selectedNode: null,
  selectedEventId: null,
  selectedEventIds: null,
  yearMode: "range",
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_END_YEAR,
  showEventLabels: false,
  projectionMode: 'global',
  scatterPlotLayerMode: 'neighbours',
  colorScatterBy: 'pub_year',
};

export const [controls, setControls] = makePersisted(
  createStore<ControlsState>(initialControls),
  { name: "controls" }
);
