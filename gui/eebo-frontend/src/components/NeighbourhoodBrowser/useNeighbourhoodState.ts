/**
 * useNeighbourhoodState.ts
 *
 * All reactive state for the NeighbourhoodBrowser lives here.
 */

import {
    createSignal,
    createMemo,
    createResource,
    onMount,
    onCleanup,
    type Setter,
} from "solid-js";

import type { SqliteEventWithNeighbours, SqliteNeighbour } from "../../types";
import { createTokenWindowResource } from "../../services/tokenWindowApi";
import { controls } from "../../state/controls.store";
import { getYearBounds, getYearFiltered } from "../../state/controls.selectors";
import { dbReady } from "../../state/tier2data.store";
import {
    eventKey,
    buildNeighbourIndex,
    buildTemporalProfile,
    toSeries,
    type NeighbourIndex,
    type TemporalProfile,
    type TemporalPoint,
} from "./neighbourUtils";

// Exported shape
export interface NeighbourhoodState {
    // Data
    yearFiltered: () => SqliteEventWithNeighbours[];
    yearBounds: () => [number, number];
    neighbourIndex: () => NeighbourIndex;
    sortedGlobalNeighbours: () => ReturnType<NeighbourIndex["values"]> extends IterableIterator<infer T> ? T[] : never;
    selectedEvent: () => { event: SqliteEventWithNeighbours; key: string } | null;
    selectedEventNeighbours: () => SqliteNeighbour[];
    selectedScoreRange: () => [number, number];
    rightPanelDocs: () => Array<{ docId: string; year?: number; token_idx: number }>;
    focusEventKeys: () => Set<string>;
    tokenTemporalProfile: () => TemporalProfile;
    toSeries: (profile: TemporalProfile, token: string) => TemporalPoint[];
    windowText: () => string | null | undefined;
    // Signals (readable)
    selectedEventId: () => string | null;
    focusToken: () => string | null;
    rightPanelEvent: () => { doc_id: string; token_idx: number } | null;
    // Loading / error state
    isLoading: () => boolean;
    // Setters
    setSelectedEventId: Setter<string | null>;
    setFocusToken: (updater: string | null | ((prev: string | null) => string | null)) => void;
    setRightPanelEvent: (
        updater:
            | { doc_id: string; token_idx: number }
            | null
            | ((prev: { doc_id: string; token_idx: number } | null) => { doc_id: string; token_idx: number } | null),
    ) => void;
    // Keyboard helpers
    eventButtonRefs: Map<string, HTMLButtonElement>;
}

// Hook
export function useNeighbourhoodState(): NeighbourhoodState {
    const [selectedEventId, setSelectedEventId] = createSignal<string | null>(null);
    const [focusToken, setFocusToken] = createSignal<string | null>(null);
    const [rightPanelEvent, setRightPanelEvent] = createSignal<{
        doc_id: string;
        token_idx: number;
    } | null>(null);

    const eventButtonRefs = new Map<string, HTMLButtonElement>();

    // Resources
    const resourceKey = () => [controls.conceptSelection[0], controls.fromYear, controls.toYear] as const;

    const [yearFilteredResource] = createResource(
        resourceKey,
        async ([concept, from, to]) => {
            if (!concept || !dbReady()) return [];
            return getYearFiltered(concept, from, to);
        },
    );

    const yearFiltered = (): SqliteEventWithNeighbours[] => yearFilteredResource() ?? [];

    const [yearBoundsResource] = createResource(
        () => controls.conceptSelection[0],
        (concept) => getYearBounds(concept),
    );

    const yearBounds = (): [number, number] => yearBoundsResource() ?? [controls.fromYear, controls.toYear];

    // Derived memos

    const neighbourIndex = createMemo<NeighbourIndex>(() =>
        buildNeighbourIndex(yearFiltered()),
    );

    const sortedGlobalNeighbours = createMemo(() =>
        [...neighbourIndex().values()].sort(
            (a, b) => b.eventCount - a.eventCount || b.meanScore - a.meanScore,
        ),
    );

    const selectedEvent = createMemo<{
        event: SqliteEventWithNeighbours;
        key: string;
    } | null>(() => {
        const id = selectedEventId();
        if (!id) return null;
        const events = yearFiltered();
        for (let idx = 0; idx < events.length; idx++) {
            const k = eventKey(events[idx], idx);
            if (k === id) return { event: events[idx], key: k };
        }
        return null;
    });

    const activeWindowEvent = createMemo(
        () => rightPanelEvent() ?? selectedEvent()?.event ?? null,
    );
    const [windowText] = createTokenWindowResource(activeWindowEvent);

    const selectedEventNeighbours = createMemo<SqliteNeighbour[]>(() => {
        const sel = selectedEvent();
        if (!sel) return [];
        return [...sel.event.neighbours].sort((a, b) => b.score - a.score);
    });

    const selectedScoreRange = createMemo<[number, number]>(() => {
        const nbs = selectedEventNeighbours();
        if (nbs.length === 0) return [0, 1];
        let min = nbs[0].score;
        let max = nbs[0].score;
        for (let i = 1; i < nbs.length; i++) {
            if (nbs[i].score < min) min = nbs[i].score;
            if (nbs[i].score > max) max = nbs[i].score;
        }
        return [min, max];
    });

    const rightPanelDocs = createMemo<
        Array<{ docId: string; year?: number; token_idx: number }>
    >(() => {
        const focusedToken = focusToken();
        if (focusedToken) {
            const summary = neighbourIndex().get(focusedToken);
            if (!summary) return [];
            return [...summary.docYears.entries()]
                .map(([docId, { year, token_idx }]) => ({ docId, year, token_idx }))
                .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
        }
        const sel = selectedEvent();
        if (sel?.event.doc_id) {
            return [{
                docId: sel.event.doc_id,
                year: sel.event.pub_year,
                token_idx: sel.event.token_idx,
            }];
        }
        return [];
    });

    const focusEventKeys = createMemo<Set<string>>(() => {
        const ft = focusToken();
        if (!ft) return new Set();
        return neighbourIndex().get(ft)?.eventKeys ?? new Set();
    });

    const tokenTemporalProfile = createMemo(() =>
        buildTemporalProfile(yearFiltered()),
    );

    // Keyboard navigation

    const selectedIndex = createMemo(() => {
        const id = selectedEventId();
        if (!id) return -1;
        return yearFiltered().findIndex((e, idx) => eventKey(e, idx) === id);
    });

    function moveSelection(delta: number) {
        const list = yearFiltered();
        const next = selectedIndex() + delta;
        if (next < 0 || next >= list.length) return;
        const key = eventKey(list[next], next);
        setSelectedEventId(key);
        queueMicrotask(() => eventButtonRefs.get(key)?.focus());
    }

    const handleKeyDown = (e: KeyboardEvent) => {
        if (selectedEventId() == null) return;
        switch (e.key) {
            case "ArrowUp":
            case "ArrowLeft":
                e.preventDefault();
                moveSelection(-1);
                break;
            case "ArrowDown":
            case "ArrowRight":
                e.preventDefault();
                moveSelection(1);
                break;
        }
    };

    onMount(() => window.addEventListener("keydown", handleKeyDown));
    onCleanup(() => window.removeEventListener("keydown", handleKeyDown));


    return {
        yearFiltered,
        yearBounds,
        neighbourIndex,
        sortedGlobalNeighbours,
        selectedEvent,
        selectedEventNeighbours,
        selectedScoreRange,
        rightPanelDocs,
        focusEventKeys,
        tokenTemporalProfile,
        toSeries,
        windowText,
        selectedEventId,
        focusToken,
        rightPanelEvent,
        isLoading: () => yearFilteredResource.loading,
        setSelectedEventId,
        setFocusToken,
        setRightPanelEvent,
        eventButtonRefs,
    };
}
