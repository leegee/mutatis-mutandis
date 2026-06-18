import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import type {
    ConceptData,
    SqliteEventWithNeighbours,
} from "../types/";


export function scanYearRange(cd: ConceptData): [number, number] {
    let min = CORPUS_END_YEAR;
    let max = CORPUS_START_YEAR;
    for (const e of cd.events) {
        if (e.pub_year === undefined) continue;
        if (e.pub_year < min) min = e.pub_year;
        if (e.pub_year > max) max = e.pub_year;
    }
    return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

export function filterByYearRange(
    events: SqliteEventWithNeighbours[],
    from: number,
    to: number,
): SqliteEventWithNeighbours[] {
    return events.filter(
        (e) => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to,
    );
}

export interface RankedToken {
    token: string;
    rank: number;
    freq: number;
    meanScore: number;
    eventCount: number;
}

export type TokenStatus = "birth" | "death" | "birth-death" | "continuation";

export type YearSlices = Map<number, RankedToken[]>;
export type SortKey = "freq" | "score";

export function buildYearSlices(
    events: SqliteEventWithNeighbours[],
    topN: number,
    window: number,
    sortKey: SortKey,
): YearSlices {
    if (events.length === 0) return new Map();

    const raw = new Map<
        number,
        Map<string, { freq: number; scoreSum: number; eventSet: Set<string> }>
    >();

    for (const e of events) {
        if (e.pub_year === undefined) continue;
        const yr = e.pub_year;
        if (!raw.has(yr)) raw.set(yr, new Map());
        const byTok = raw.get(yr)!;
        const seenThisEvent = new Set<string>();

        for (const nb of e.neighbours) {
            let rec = byTok.get(nb.token);
            if (!rec) {
                rec = { freq: 0, scoreSum: 0, eventSet: new Set() };
                byTok.set(nb.token, rec);
            }
            rec.freq += 1;
            rec.scoreSum += nb.score;
            if (!seenThisEvent.has(nb.token)) {
                rec.eventSet.add(String(e.event_id ?? `idx`));
                seenThisEvent.add(nb.token);
            }
        }
    }

    const years = [...raw.keys()].sort((a, b) => a - b);
    const smoothed = new Map<
        number,
        Map<string, { freq: number; scoreSum: number; eventCount: number }>
    >();

    for (const yr of years) {
        const merged = new Map<
            string,
            { freq: number; scoreSum: number; eventCount: number }
        >();
        for (let dy = -window; dy <= window; dy++) {
            const src = raw.get(yr + dy);
            if (!src) continue;
            for (const [tok, rec] of src) {
                let m = merged.get(tok);
                if (!m) {
                    m = { freq: 0, scoreSum: 0, eventCount: 0 };
                    merged.set(tok, m);
                }
                m.freq += rec.freq;
                m.scoreSum += rec.scoreSum;
                m.eventCount += rec.eventSet.size;
            }
        }
        smoothed.set(yr, merged);
    }

    const slices: YearSlices = new Map();

    for (const yr of years) {
        const merged = smoothed.get(yr)!;
        const sorted = [...merged.entries()]
            .sort((a, b) => {
                const va =
                    sortKey === "freq"
                        ? a[1].freq
                        : a[1].freq > 0
                            ? a[1].scoreSum / a[1].freq
                            : 0;
                const vb =
                    sortKey === "freq"
                        ? b[1].freq
                        : b[1].freq > 0
                            ? b[1].scoreSum / b[1].freq
                            : 0;
                return vb - va;
            })
            .slice(0, topN);

        slices.set(
            yr,
            sorted.map(([token, rec], rank) => ({
                token,
                rank,
                freq: rec.freq,
                meanScore: rec.freq > 0 ? rec.scoreSum / rec.freq : 0,
                eventCount: rec.eventCount,
            })),
        );
    }

    return slices;
}
