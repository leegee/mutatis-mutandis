import { createStore } from "solid-js/store";
import { makePersisted } from "@solid-primitives/storage";
import type { LabelPoint } from "../components/ScatterPlot/types";

export interface LabelStoreState {
  labels: LabelPoint[];
  minCentroidDistance: number;
}

const initialLabelState = {
  labels: [] as LabelPoint[],
  minCentroidDistance: 0,
}

export const [labelState, setLabelState] = makePersisted(
  createStore<LabelStoreState>(initialLabelState),
  { name: "labels" }
);
