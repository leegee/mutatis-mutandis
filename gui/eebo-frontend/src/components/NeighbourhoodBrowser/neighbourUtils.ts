/**
 * neighbourUtils.ts
 *
 * Pure helpers for the NeighbourhoodBrowser feature.
 * No SolidJS dependency — all functions are independently testable.
 */

import type { SqliteEventWithNeighbours } from "../../types";

export interface NeighbourSummary {
  token: string;
  token_idx: string;
  occurrenceCount: number;
  eventCount: number;
  meanScore: number;
  docIds: Set<string>;
  docYears: Map<string, { year: number | undefined; token_idx: number }>;
  eventKeys: Set<string>;
}

export type NeighbourIndex = Map<string, NeighbourSummary>;

export type TemporalProfile = Map<string, Map<number, Set<string>>>;

export interface TemporalPoint {
  year: number;
  value: number;
}

// Event identity
export function eventKey(event: SqliteEventWithNeighbours, idx: number): string {
  return event.event_id !== undefined ? String(event.event_id) : `idx:${ idx }`;
}

// Index construction
export function buildNeighbourIndex(events: SqliteEventWithNeighbours[]): NeighbourIndex {
  const index: NeighbourIndex = new Map();

  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const key = eventKey(event, idx);
    const seenInEvent = new Set<string>();

    for (const nb of event.neighbours) {
      let summary = index.get(nb.token);

      if (!summary) {
        summary = {
          token: nb.token,
          token_idx: nb.token_idx,
          occurrenceCount: 0,
          eventCount: 0,
          meanScore: 0,
          docIds: new Set(),
          docYears: new Map<string, { year: number | undefined; token_idx: number }>,
          eventKeys: new Set(),
        };
        index.set(nb.token, summary);
      }

      summary.occurrenceCount += 1;

      if (!seenInEvent.has(nb.token)) {
        summary.eventCount += 1;
        summary.eventKeys.add(key);
        seenInEvent.add(nb.token);
      }

      summary.meanScore += nb.score;

      if (event.doc_id) {
        summary.docIds.add(event.doc_id);
        if (!summary.docYears.has(event.doc_id)) {
          summary.docYears.set(event.doc_id, { year: event.pub_year, token_idx: event.token_idx });
        }
      }
    }
  }

  for (const s of index.values()) {
    if (s.occurrenceCount > 0) s.meanScore /= s.occurrenceCount;
  }

  return index;
}

// Score → visual opacity
export function scoreToOpacity(
  score: number,
  scoreMin: number,
  scoreMax: number,
  minOp = 0.5,
  maxOp = 1.0,
): number {
  const range = scoreMax - scoreMin;
  if (range < 1e-9) return maxOp;
  return minOp + ((score - scoreMin) / range) * (maxOp - minOp);
}

// Temporal profile
export function buildTemporalProfile(
  events: SqliteEventWithNeighbours[],
): TemporalProfile {
  const map: TemporalProfile = new Map();

  for (let idx = 0; idx < events.length; idx++) {
    const e = events[idx];
    const year = e.pub_year;
    if (year === undefined) continue;
    const key = eventKey(e, idx);

    for (const nb of e.neighbours) {
      let byYear = map.get(nb.token);
      if (!byYear) {
        byYear = new Map();
        map.set(nb.token, byYear);
      }
      let set = byYear.get(year);
      if (!set) {
        set = new Set();
        byYear.set(year, set);
      }
      set.add(key);
    }
  }

  return map;
}

export function toSeries(
  profile: TemporalProfile,
  token: string,
): TemporalPoint[] {
  const byYear = profile.get(token);
  if (!byYear) return [];
  return [...byYear.entries()]
    .map(([year, set]) => ({ year, value: set.size }))
    .sort((a, b) => a.year - b.year);
}

