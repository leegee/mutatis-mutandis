// src/tier2/neighbourhood-query.ts

import { query } from "@solidjs/router";
import { loadNeighbourhoodData } from "./neighbourhood";

export const getNeighbourhoodData = query(async (concept: string, fromYear: number, toYear: number) => {
	"use server";

	const started = performance.now();

	console.log("[getNeighbourhoodData] start with", fromYear, toYear);

	const result = await loadNeighbourhoodData(concept, fromYear, toYear);

	console.log(`[getNeighbourhoodData] server query: ${(performance.now() - started).toFixed(1)} ms`);
	console.log(`[getNeighbourhoodData] result ${result.events.length}`);

	return result;
}, "neighbourhood-data");
