import type { NeighbourhoodData, NeighbourhoodEvent } from "~/types/neighbourhood";
import { get_connection } from "../db";

interface EventRow {
	event_id: string;
	doc_id: string;
	token: string;
	token_idx: number;
	pub_year: number;
}

interface NeighbourRow {
	seed_event_id: string;
	neighbour_event_id: string;
	score: number;
	doc_id: string;
	token: string;
	token_idx: number;
	pub_year: number | null;
}

export async function loadNeighbourhoodData(
	concept: string,
	fromYear: number,
	toYear: number,
): Promise<NeighbourhoodData> {
	if (!concept) {
		return {
			events: [],
			yearBounds: [fromYear, toYear],
		};
	}

	const db = get_connection();

	const eventRows = await db<EventRow[]>`
		SELECT
			e.event_id,
			e.doc_id,
			e.token,
			e.token_idx,
			e.pub_year
		FROM tier2.event_field ef
		JOIN events e
			ON e.event_id = ef.event_id
		WHERE ef.concept = ${concept}
		  AND e.pub_year IS NOT NULL
		  AND e.pub_year BETWEEN ${fromYear} AND ${toYear}
		ORDER BY e.pub_year, e.event_id
	`;

	const neighbourRows = await db<NeighbourRow[]>`
		SELECT
			ne.seed_event_id,
			ne.neighbour_event_id,
			ne.score,
			e.doc_id,
			e.token,
			e.token_idx,
			e.pub_year
		FROM (
			SELECT e.event_id
			FROM tier2.event_field ef
			JOIN events e
				ON e.event_id = ef.event_id
			WHERE ef.concept = ${concept}
			AND e.pub_year IS NOT NULL
			AND e.pub_year BETWEEN ${fromYear} AND ${toYear}
		) seeds
		JOIN tier2.neighbour_edges ne
			ON ne.seed_event_id = seeds.event_id
		JOIN events e
			ON e.event_id = ne.neighbour_event_id
		ORDER BY ne.seed_event_id, ne.score DESC
	`;

	const events: NeighbourhoodEvent[] = eventRows.map((row) => ({
		eventId: String(row.event_id),
		docId: row.doc_id,
		token: row.token,
		tokenIdx: Number(row.token_idx),
		pubYear: Number(row.pub_year),
		neighbours: [],
	}));

	const eventsById = new Map(events.map((event) => [event.eventId, event]));

	for (const row of neighbourRows) {
		const event = eventsById.get(row.seed_event_id);

		if (!event) {
			continue;
		}

		event.neighbours.push({
			eventId: String(row.neighbour_event_id),
			docId: row.doc_id,
			token: row.token,
			tokenIdx: String(row.token_idx),
			pubYear: Number(row.pub_year),
			score: Number(row.score),
		});
	}

	const years = eventRows.map((row) => Number(row.pub_year)).filter(Number.isFinite);

	const yearBounds: [number, number] = years.length ? [Math.min(...years), Math.max(...years)] : [fromYear, toYear];

	return {
		events,
		yearBounds,
	};
}
