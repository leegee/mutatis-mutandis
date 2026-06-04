import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { setControls, type ViewMode, type YearMode } from "./controls.store";

export const controlsActions = {
    setConcept(concept: string) {
        setControls({
            concept,
            selectedNode: null,
        });
    },

    setSelectedNode(id: string | null) {
        setControls("selectedNode", (prev) => (prev === id ? null : id));
    },

    setViewMode(mode: ViewMode) {
        setControls({
            viewMode: mode,
            selectedNode: null,
        });
    },

    setMaxHubs(maxHubs: number) {
        setControls("maxHubs", maxHubs);
    },

    setTopN(topN: number) {
        setControls("topN", topN);
    },

    setHubSpread(v: number) {
        setControls("hubSpread", v);
    },

    setMinSimilarity(v: number) {
        setControls("minSimilarity", v);
    },

    setYearMode(mode: YearMode, bounds: [number, number]) {
        const [min, max] = bounds;
        const mid = Math.floor((min + max) / 2);

        setControls({
            yearMode: mode,
            fromYear: mode === "single" ? mid : min,
            toYear: mode === "single" ? mid : max,
        });
    },

    setSingleYear(v: number) {
        setControls({
            fromYear: v,
            toYear: v,
        });
    },

    stepYear(delta: number) {
        setControls((s) => {
            const v = Math.min(
                CORPUS_END_YEAR,
                Math.max(CORPUS_START_YEAR, s.fromYear + delta)
            );

            return {
                fromYear: v,
                toYear: v,
            };
        });
    },

    setRange(from: number, to: number) {
        setControls({
            fromYear: from,
            toYear: to,
        });
    },

    clearSelection() {
        setControls("selectedNode", null);
    },
};
