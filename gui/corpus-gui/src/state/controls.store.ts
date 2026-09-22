// src/state/controls.store.ts

import { createStore } from "solid-js/store";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

export type ControlsState = {
	conceptSelection: string[];
	fromYear: number;
	toYear: number;
};

const initialControls: ControlsState = {
	conceptSelection: ["WHITE"],
	fromYear: CORPUS_START_YEAR,
	toYear: CORPUS_END_YEAR,
};

export const [controls, setControls] = createStore<ControlsState>(initialControls);
