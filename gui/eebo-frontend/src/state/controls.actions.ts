import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { setControls, type ViewMode, type YearMode } from "./controls.store";

const clampYear = (y: number) =>
    Math.min(CORPUS_END_YEAR, Math.max(CORPUS_START_YEAR, y));

const normalizeRange = (from: number, to: number) => {
    const a = clampYear(from);
    const b = clampYear(to);
    return {
        fromYear: Math.min(a, b),
        toYear: Math.max(a, b),
    };
};

export const controlsActions = {
    setConcept(concept: string) {
        console.log('[actions] setConcept', concept);
        setControls({
            concept,
            selectedNode: null,
        });
    },

    // Sets concept if the concept selection is just one - deselects the current node
    setConceptSelection(
        conceptSelection: string[] | ((prev: string[]) => string[])
    ) {
        setControls((prev) => {
            const nextConceptSelection = typeof conceptSelection === "function" ? conceptSelection(prev.conceptSelection) : conceptSelection;
            return {
                ...prev,
                conceptSelection: nextConceptSelection,
                concept: nextConceptSelection.length === 1 ? nextConceptSelection[0] : prev.concept, // null to clear it
                selectedNode: null,
            };
        });
    },

    setSelectedNode(id: string | null) {
        setControls("selectedNode", (prev) => (prev === id ? null : id));
    },

    setSelectedEventId(id: string | null) {
        setControls("selectedEventId", id);
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

    // YEAR API
    setYearMode(mode: YearMode, bounds: [number, number]) {
        const [min, max] = bounds;

        if (mode === "single") {
            const mid = clampYear(Math.floor((min + max) / 2));

            setControls({
                yearMode: "single",
                fromYear: mid,
                toYear: mid,
            });
            return;
        }

        setControls({
            yearMode: "range",
            fromYear: clampYear(min),
            toYear: clampYear(max),
        });
    },

    setSingleYear(year: number) {
        const v = clampYear(year);

        setControls({
            yearMode: "single",
            fromYear: v,
            toYear: v,
        });
    },

    setRange(from: number, to: number) {
        const { fromYear, toYear } = normalizeRange(from, to);

        setControls({
            yearMode: "range",
            fromYear,
            toYear,
        });
    },

    setAllYears() {
        setControls({
            yearMode: "range",
            fromYear: CORPUS_START_YEAR,
            toYear: CORPUS_END_YEAR,
        });
    },

    stepYear(delta: number) {
        setControls((s) => {
            const base =
                s.yearMode === "single" ? s.fromYear : s.toYear;

            const next = clampYear(base + delta);

            if (s.yearMode === "single") {
                return {
                    yearMode: "single",
                    fromYear: next,
                    toYear: next,
                };
            }

            // range mode: expand in direction
            const { fromYear, toYear } =
                delta < 0
                    ? normalizeRange(next, s.toYear)
                    : normalizeRange(s.fromYear, next);

            return {
                yearMode: "range",
                fromYear,
                toYear,
            };
        });
    },

    clearSelection() {
        setControls("selectedNode", null);
    },
};
