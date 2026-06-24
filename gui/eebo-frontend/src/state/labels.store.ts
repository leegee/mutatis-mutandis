import { createStore } from "solid-js/store";
import type { LabelDataset } from "../components/ScatterPlot/types";

export interface LabelStoreState {
  labelDataset?: LabelDataset;
}

export const [labelState, setLabelState] = createStore<LabelStoreState>({
  labelDataset: undefined,
});
