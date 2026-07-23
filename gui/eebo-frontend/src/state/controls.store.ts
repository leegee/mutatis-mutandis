// src/state/controls.store.ts

import { createStore } from "solid-js/store";
import { makePersisted } from "@solid-primitives/storage";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import type { ViewMode, YearMode } from "../types";
import type { PointData } from "../components/ScatterPlot/types";

export const MAX_TOP_N = 1000;

export type ProjectionModeType = "local" | "global"
export type ColorScatterByType = 'pub_year' | "cluster_label" | "doc_id"

export type ControlsState = {
  concept: string;
  bfsOpacity: number;
  neighbourOpacity: number; // = alpha in range 0 - 255
  conceptSelection: string[];
  showNeighbours: boolean;
  showClusterCentroids: boolean;
  colorScatterBy: ColorScatterByType;
  viewMode: ViewMode;
  maxHubs: number;
  topN: number;
  minSimilarity: number;
  hubSpread: number;
  selectedNode: string | null;
  selectedEventId: string | null;
  selectedEventIds: Set<string>;
  selectedPoints: PointData[],
  yearMode: YearMode;
  fromYear: number;
  toYear: number;
  showEventLabels: boolean;
  projectionMode: ProjectionModeType;
  authorMatch: string;
};

const initialControls: ControlsState = {
  bfsOpacity: 3,
  neighbourOpacity: 100,
  concept: "LIBERTY",
  conceptSelection: ["LIBERTY"],
  viewMode: "events",
  maxHubs: 50,
  topN: MAX_TOP_N,
  minSimilarity: 0.5,
  hubSpread: 1,
  selectedNode: null,
  selectedEventId: null,
  selectedEventIds: new Set(),
  selectedPoints: [],
  yearMode: "range",
  fromYear: CORPUS_START_YEAR,
  toYear: CORPUS_END_YEAR,
  showEventLabels: false,
  projectionMode: 'global',
  showNeighbours: true,
  showClusterCentroids: true,
  colorScatterBy: 'pub_year',
  authorMatch: '',
};

export const [controls, setControls] = makePersisted(
  createStore<ControlsState>(initialControls),
  { name: "controls" }
);
