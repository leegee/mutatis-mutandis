import { createStore } from "solid-js/store";
import { makePersisted } from "@solid-primitives/storage";
import type { LabelPoint } from "../components/ScatterPlot/types";

type ConceptToLabelType = Record<string, LabelPoint[]>;

export interface LabelStoreState {
  labels: ConceptToLabelType;
  minCentroidDistance: number;
}

const initialLabelState = {
  labels: {},
  minCentroidDistance: 0,
}

export const [labelState, setLabelState] = makePersisted(
  createStore<LabelStoreState>(initialLabelState),
  { name: "labels" }
);
