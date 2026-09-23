// year-query.ts

import { query } from "@solidjs/router";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "~/corpus_config";
import { get_connection } from "../db";

export interface YearBucket {
	year: number;
	count: number;
}

export const getYearBuckets = query(async (concept: string): Promise<YearBucket[]> => {
	"use server";

	const db = get_connection();

	const rows = await db<{ pub_year: number; count: number }[]>`
		SELECT
			e.pub_year,
			COUNT(*) AS count
		FROM tier2.event_field ef
		JOIN events e ON e.event_id = ef.event_id
		WHERE ef.concept = ${concept}
		  AND e.pub_year IS NOT NULL
		GROUP BY e.pub_year
		ORDER BY e.pub_year ASC
	`;

	const counts = new Map<number, number>();

	for (const row of rows) {
		counts.set(Number(row.pub_year), Number(row.count));
	}

	const buckets: YearBucket[] = [];

	for (let year = CORPUS_START_YEAR; year <= CORPUS_END_YEAR; year++) {
		buckets.push({
			year,
			count: counts.get(year) ?? 0,
		});
	}

	return buckets;
}, "year-buckets");
