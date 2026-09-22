import { describe, expect, test } from "bun:test";
import { loadNeighbourhoodData } from "./neighbourhood";

describe("Tier 2 neighbourhood data", () => {
	test("loads WHITE events and neighbours", async () => {
		const result = await loadNeighbourhoodData("WHITE", 1500, 1549);

		expect(result.events.length).toBeGreaterThan(0);
		expect(result.yearBounds[0]).toBeLessThanOrEqual(result.events[0].pubYear);
		expect(result.yearBounds[1]).toBeGreaterThanOrEqual(result.events.at(-1)!.pubYear);

		for (const event of result.events) {
			expect(event.pubYear).toBeGreaterThanOrEqual(1500);
			expect(event.pubYear).toBeLessThanOrEqual(1549);
			expect(event.eventId).toBeGreaterThan(0);
			expect(event.docId).not.toBe("");
			expect(event.token).not.toBe("");
		}
	});

	test("events are ordered by publication year and event ID", async () => {
		const result = await loadNeighbourhoodData("WHITE", 1500, 1549);

		for (let i = 1; i < result.events.length; i += 1) {
			const previous = result.events[i - 1];
			const current = result.events[i];

			expect(
				current.pubYear > previous.pubYear ||
					(current.pubYear === previous.pubYear && current.eventId >= previous.eventId),
			).toBe(true);
		}
	});

	test("neighbours have valid event identities and scores", async () => {
		const result = await loadNeighbourhoodData("WHITE", 1500, 1549);

		for (const event of result.events) {
			for (const neighbour of event.neighbours) {
				expect(neighbour.eventId).toBeGreaterThan(0);
				expect(neighbour.score).toBeTypeOf("number");
				expect(Number.isFinite(neighbour.score)).toBe(true);
			}
		}
	});

	test("empty concept returns empty data without querying events", async () => {
		const result = await loadNeighbourhoodData("", 1500, 1549);

		expect(result).toEqual({
			events: [],
			yearBounds: [1500, 1549],
		});
	});

	test("a range with no matching concept events returns empty events", async () => {
		const result = await loadNeighbourhoodData("WHITE", 1, 2);

		expect(result.events).toEqual([]);
		expect(result.yearBounds[0]).toBeLessThan(result.yearBounds[1]);
	});

	test("returns neighbourhood data", async () => {
		const started = performance.now();

		const result = await loadNeighbourhoodData("WHITE", 1500, 1549);

		const neighbourCount = result.events.reduce((total, event) => total + event.neighbours.length, 0);

		const json = JSON.stringify(result);

		console.log(
			`events=${result.events.length}, ` +
				`neighbours=${neighbourCount}, ` +
				`JSON=${(json.length / 1024 / 1024).toFixed(2)} MiB, ` +
				`time=${(performance.now() - started).toFixed(1)} ms`,
		);

		expect(result.events.length).toBeGreaterThan(0);
	});
});
