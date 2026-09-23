import { createStore } from "solid-js/store";
import type { YearMode } from "~/types/controls";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

export type ControlsState = {
	conceptSelection: string[];
	yearMode: YearMode;
	fromYear: number;
	toYear: number;
	selectedEventIds: Set<string>;
};

const initialControls: ControlsState = {
	conceptSelection: ["WHITE"],
	yearMode: "single",
	fromYear: CORPUS_START_YEAR,
	toYear: CORPUS_END_YEAR,
	selectedEventIds: new Set<string>(),
};

export const [controls, setControls] = createStore<ControlsState>(initialControls);
