/**
 * useNeighbourhoodState.ts
 *
 * Client-side loading and selection state for the neighbourhood browser.
 * PostgreSQL access is confined to getNeighbourhoodData().
 */

import { createAsync } from "@solidjs/router";
import { createSignal, onCleanup, onMount, type Setter } from "solid-js";
import { getNeighbourhoodData } from "~/server/tier2/neighbourhood-query";
import { controls } from "~/state/controls.store";
import type { NeighbourhoodData } from "~/types/neighbourhood";

export interface NeighbourhoodState {
	data: () => NeighbourhoodData;
	selectedEventId: () => number | null;
	isLoading: () => boolean;
	error: () => string | null;
	setSelectedEventId: Setter<number | null>;
	eventButtonRefs: Map<string, HTMLButtonElement>;
}

const emptyData = (fromYear: number, toYear: number): NeighbourhoodData => ({
	events: [],
	yearBounds: [fromYear, toYear],
});

export function useNeighbourhoodState(): NeighbourhoodState {
	const [selectedEventId, setSelectedEventId] = createSignal<number | null>(null);

	const [error, setError] = createSignal<string | null>(null);

	const eventButtonRefs = new Map<string, HTMLButtonElement>();

	const resourceKey = () => [controls.conceptSelection[0], controls.fromYear, controls.toYear] as const;

	const resource = createAsync(
		() => {
			const [concept, fromYear, toYear] = resourceKey();

			if (!concept) {
				setError(null);
				setSelectedEventId(null);
				return Promise.resolve(emptyData(fromYear, toYear));
			}

			return getNeighbourhoodData(concept, fromYear, toYear).catch((err: any) => {
				const message = err instanceof Error ? err.message : String(err);

				setError(message);
				setSelectedEventId(null);

				return emptyData(fromYear, toYear);
			});
		},
		{
			initialValue: emptyData(controls.fromYear, controls.toYear),
		},
	);

	const data = (): NeighbourhoodData => {
		const value = resource();

		if (value) {
			return value;
		}

		return emptyData(controls.fromYear, controls.toYear);
	};

	const events = () => data().events;

	const selectedIndex = () => {
		const id = selectedEventId();

		if (id === null) {
			return -1;
		}

		return events().findIndex((event) => event.eventId === id);
	};

	function moveSelection(delta: number) {
		const list = events();
		const current = selectedIndex();
		const next = current + delta;

		if (next < 0 || next >= list.length) {
			return;
		}

		const event = list[next];

		setSelectedEventId(event.eventId);

		queueMicrotask(() => {
			eventButtonRefs.get(String(event.eventId))?.focus();
		});
	}

	const handleKeyDown = (event: KeyboardEvent) => {
		if (selectedEventId() === null) {
			return;
		}

		switch (event.key) {
			case "ArrowUp":
			case "ArrowLeft":
				event.preventDefault();
				moveSelection(-1);
				break;

			case "ArrowDown":
			case "ArrowRight":
				event.preventDefault();
				moveSelection(1);
				break;
		}
	};

	onMount(() => {
		window.addEventListener("keydown", handleKeyDown);
		onCleanup(() => {
			window.removeEventListener("keydown", handleKeyDown);
		});
	});

	return {
		data,
		selectedEventId,
		isLoading: () => false, // resource.loading,
		error,
		setSelectedEventId,
		eventButtonRefs,
	};
}
