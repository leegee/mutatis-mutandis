// src/state/selection.ts

import { createSignal } from "solid-js";

export const [selectedConcept, setSelectedConcept] = createSignal<string | null>(null);

export const [selectedSlice, setSelectedSlice] = createSignal<string | null>(null);

export const [selectedEventId, setSelectedEventId] = createSignal<string | null>(null);

export const [hoveredEventId, setHoveredEventId] = createSignal<string | null>(null);