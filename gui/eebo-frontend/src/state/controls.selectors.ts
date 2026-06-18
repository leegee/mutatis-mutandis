/**
 * src/state/selectors.ts
 *
 * All data-fetching functions are async — queries cross the main→worker
 * boundary.  Use createResource() in components, not createMemo().
 *
 * Exception: filterEvents and yearBoundsFrom are synchronous utilities
 * for callers that already hold data in memory.
 */

import { CORPUS_START_YEAR, CORPUS_END_YEAR } from "../corpus_config";
import { controls } from "./controls.store";
import { queryYearBounds, queryEventsByConcept, listConcepts, queryNEvents, queryYearCounts } from "../services/db";
import { filterByYearRange, scanYearRange } from "../lib/yearUtils";
import type { ConceptData, SqliteEventWithNeighbours } from "../types";

export async function getConcepts(): Promise<string[]> {
  return listConcepts();
}

export async function getYearBounds(
  concept?: string,
): Promise<[number, number]> {
  const c = concept ?? controls.concept;
  if (!c) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
  return (await queryYearBounds(c)) ?? [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

export async function getYearFiltered(
  concept?: string,
  fromYear?: number,
  toYear?: number,
): Promise<SqliteEventWithNeighbours[]> {
  console.debug(`[getYearFiltered] ${ concept } ${ fromYear } ${ toYear }`);
  const c = concept ?? controls.concept;
  const fy = fromYear ?? controls.fromYear;
  const ty = toYear ?? controls.toYear;
  if (!c) return [];
  return queryEventsByConcept(c, fy, ty);
}

export async function totalEventsForConcept(concept?: string): Promise<number> {
  const c = concept ?? controls.concept;
  if (!c) return 0;
  return await queryNEvents(c);
}

/** Synchronous — for callers that already hold a ConceptData object. */
export function yearBoundsFrom(conceptData: ConceptData): [number, number] {
  return scanYearRange(conceptData);
}

/** Synchronous — for callers that already hold a ConceptEvent[] in memory. */
export function filterEvents(
  events: SqliteEventWithNeighbours[],
  from: number,
  to: number,
  bounds?: [number, number],
): SqliteEventWithNeighbours[] {
  if (bounds) {
    const [min, max] = bounds;
    if (from <= min && to >= max) return events;
  }
  return filterByYearRange(events, from, to);
}


/** Used by YearTimeline */
interface YearBucket {
  year: number;
  count: number;
}

export async function getYearBuckets(
  concept?: string,
): Promise<YearBucket[]> {
  const c = concept ?? controls.concept;
  if (!c) return [];

  const [[minYear, maxYear], tally] = await Promise.all([
    getYearBounds(c),
    queryYearCounts(c),
  ]);

  const buckets: YearBucket[] = [];
  for (let year = minYear; year <= maxYear; year++) {
    buckets.push({ year, count: tally.get(year) ?? 0 });
  }
  return buckets;
}
