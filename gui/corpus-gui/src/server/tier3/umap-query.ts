// tier3/umap-query.ts

import { query } from "@solidjs/router";
import type { YearMode } from "~/types/controls";
import { loadUmapData } from "./umap";

export const getUmapData = query(async (concept: string, yearMode: YearMode, fromYear: number, toYear: number, seedsOnly: boolean) => {
	"use server";

	const started = performance.now();

	console.log("[getUmapData] start with", concept, "seeds only =", seedsOnly);

	const result = await loadUmapData(concept, yearMode, fromYear, toYear);

	if (seedsOnly) { result.points = result.points.filter( (point) => point.isSeed, ); }

	console.log(`[getUmapData] server query: ${(performance.now() - started).toFixed(1)} ms`);
	console.log(`[getUmapData] result ${result.points.length}`);

	return result;
}, "umap-data");
