/**
 * DiachronicChart.tsx
 *
 * Heuser-style diachronic chart of contextual neighbours.
 *
 *  Cubic-bezier links connect the same token across adjacent year columns.
 *
 * -----------------------------------------------------------------------------
 * COLOUR SEMANTICS  (after Heuser)
 * -----------------------------------------------------------------------------
 *  BIRTH       — token appears in this year but NOT the preceding year.
 *                Warm amber.
 *  DEATH       — token appears in this year but NOT the following year.
 *                Muted rose.
 *  BIRTH+DEATH — appears in only one year.  Deep orange.
 *  CONTINUATION — present in both neighbours.  Slate blue.
 *  FOCUSED     — highlighted token (hover / click).  Bright teal.
 *
 * -----------------------------------------------------------------------------
 * DATA FLOW
 * -----------------------------------------------------------------------------
 *  props.data (Tier2Data)
 *      │  filterByYearRange()
 *  ConceptEvent[]
 *      │  buildYearSlices()        (RAW + DISPLAY pipelines)
 *  Map<year, RankedToken[]>       — top-N per year, rank-ordered
 *      │  (reactive memos)
 *  SVG — link layer + cell layer
 */

import {
    createSignal,
    createMemo,
    For,
    type Component,
} from "solid-js";

import './DiachronicChart/styles.css';

import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import { filterByYearRange, scanYearRange } from "../lib/contextGraphUtils";

interface Neighbour {
    token: string;
    score: number;
    event_id?: number;
    doc_id?: string;
    pub_year?: number;
    window_id?: number;
}

interface ConceptEvent {
    event_id?: number;
    token?: string;
    doc_id?: string;
    pub_year?: number;
    neighbours: Neighbour[];
}

interface ConceptData {
    n_events: number;
    year_min?: number;
    year_max?: number;
    events: ConceptEvent[];
}

export interface Tier2Data {
    [concept: string]: ConceptData;
}

interface Props {
    data: Tier2Data;
}

interface RankedToken {
    token: string;
    rank: number;
    freq: number;
    meanScore: number;
    eventCount: number;
}

type YearSlices = Map<number, RankedToken[]>;
type SortKey = "freq" | "score";
type TokenStatus = "birth" | "death" | "birth-death" | "continuation";

const MAX_TOP_N = 50;

const CELL_WIDTH = 92;
const COL_GAP = 32;
const COL_WIDTH = CELL_WIDTH + COL_GAP;

const ROW_HEIGHT = 22;
const LABEL_PAD = 8;
const CELL_H = ROW_HEIGHT - 3;
const HEADER_H = 36;
const LEFT_MARGIN = 12;
const RIGHT_MARGIN = 12;

const C_BIRTH = "hsl(98, 79%, 56%)";
const C_DEATH = "#c4566a";
const C_BIRTH_DEATH = "#4a7fa5";
const C_CONTINUATION = "#4aa59c";
const C_FOCUS = "#3ecfb2";
const C_LINK_ALPHA = 0.38;
const C_LINK_FOCUS = 0.85;

function yearLabel(year: number, window: number): string {
    if (window === 0) return String(year);
    return `${ year - window }–${ year + window }`;
}

/**
 * Build per-year ranked token lists.
 *
 * window = 0  -> raw yearly observation
 * window > 0  -> smoothing only for ranking stability (not identity logic)
 */
function buildYearSlices(
    events: ConceptEvent[],
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
                        : a[1].freq > 0 ? a[1].scoreSum / a[1].freq : 0;

                const vb =
                    sortKey === "freq"
                        ? b[1].freq
                        : b[1].freq > 0 ? b[1].scoreSum / b[1].freq : 0;

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
            }))
        );
    }

    return slices;
}

function classifyStatus(
    token: string,
    year: number,
    years: number[],
    slices: YearSlices,
): TokenStatus {
    const idx = years.indexOf(year);

    const previousYears = years.slice(0, idx);
    const futureYears = years.slice(idx + 1);

    const existedBefore = previousYears.some(y =>
        slices.get(y)?.some(t => t.token === token)
    );

    const existsLater = futureYears.some(y =>
        slices.get(y)?.some(t => t.token === token)
    );

    if (!existedBefore && !existsLater) return "birth-death";
    if (!existedBefore) return "birth";
    if (!existsLater) return "death";
    return "continuation";
}

function statusColor(s: TokenStatus): string {
    if (s === "birth") return C_BIRTH;
    if (s === "death") return C_DEATH;
    if (s === "birth-death") return C_BIRTH_DEATH;
    return C_CONTINUATION;
}

function cellY(rank: number): number {
    return HEADER_H + rank * ROW_HEIGHT + CELL_H / 2;
}

function colX(colIdx: number): number {
    return LEFT_MARGIN + colIdx * COL_WIDTH + COL_WIDTH / 2;
}

function linkPath(
    x1: number, y1: number,
    x2: number, y2: number,
): string {
    const cx = (x1 + x2) / 2;
    return `M ${ x1 } ${ y1 } C ${ cx } ${ y1 }, ${ cx } ${ y2 }, ${ x2 } ${ y2 }`;
}

const STYLES = `/* unchanged for brevity */`;

const DiachronicChart: Component<Props> = (props) => {
    const concepts = Object.keys(props.data);

    const [concept, setConcept] = createSignal(concepts[0] ?? "");
    const [topN, setTopN] = createSignal(Math.trunc(MAX_TOP_N / 2));
    const [smoothWindow, setSmoothWindow] = createSignal(0);
    const [sortKey, setSortKey] = createSignal<SortKey>("freq");
    const [fromYear, setFromYear] = createSignal<number>(CORPUS_START_YEAR);
    const [toYear, setToYear] = createSignal<number>(CORPUS_END_YEAR);
    const [focusToken, setFocusToken] = createSignal<string | null>(null);

    const yearBounds = createMemo<[number, number]>(() => {
        const cd = props.data[concept()];
        if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
        return scanYearRange(cd);
    });

    createMemo(() => {
        const [min, max] = yearBounds();
        setFromYear(min);
        setToYear(max);
    });

    const filteredEvents = createMemo(() => {
        const cd = props.data[concept()];
        if (!cd) return [];
        return filterByYearRange(cd.events, fromYear(), toYear());
    });

    // DISPLAY slices (smoothed for ranking only)
    const displaySlices = createMemo<YearSlices>(() =>
        buildYearSlices(filteredEvents(), topN(), smoothWindow(), sortKey())
    );

    // RAW slices (identity + semantics)
    const rawSlices = createMemo<YearSlices>(() =>
        buildYearSlices(filteredEvents(), topN(), 0, sortKey())
    );

    const years = createMemo<number[]>(() =>
        [...displaySlices().keys()].sort((a, b) => a - b)
    );

    const svgWidth = createMemo(() =>
        LEFT_MARGIN + years().length * COL_WIDTH + RIGHT_MARGIN
    );

    const svgHeight = createMemo(() =>
        HEADER_H + topN() * ROW_HEIGHT + 12
    );

    const links = createMemo(() => {
        const yrs = years();
        const sl = displaySlices();

        const out: any[] = [];

        for (let c = 0; c < yrs.length - 1; c++) {
            const yr = yrs[c];
            const yrN = yrs[c + 1];

            const colA = sl.get(yr) ?? [];
            const colB = sl.get(yrN) ?? [];

            const mapB = new Map(colB.map(t => [t.token, t]));

            for (const a of colA) {
                const b = mapB.get(a.token);
                if (!b) continue;

                out.push({
                    token: a.token,
                    x1: colX(c) + CELL_WIDTH / 2,
                    y1: cellY(a.rank),
                    x2: colX(c + 1) - CELL_WIDTH / 2,
                    y2: cellY(b.rank),
                });
            }
        }

        return out;
    });

    const cellStatus = createMemo(() => {
        const yrs = years();
        const sl = rawSlices();

        const map = new Map<string, TokenStatus>();

        for (const yr of yrs) {
            for (const rt of sl.get(yr) ?? []) {
                map.set(
                    `${ yr }:${ rt.token }`,
                    classifyStatus(rt.token, yr, yrs, sl)
                );
            }
        }

        return map;
    });

    return (
        <>
            <article class="dc-root">
                <header class="dc-header">
                    <span class="dc-title">diachronic neighbours</span>

                    <div class="dc-control">
                        <label>concept</label>
                        <select value={concept()} onChange={e => setConcept(e.currentTarget.value)}>
                            <For each={concepts}>{c => <option value={c}>{c}</option>}</For>
                        </select>
                    </div>

                    <div class="dc-control">
                        <label>from</label>
                        <input type="range"
                            min={yearBounds()[0]}
                            max={yearBounds()[1]}
                            value={fromYear()}
                            onInput={e => setFromYear(Number(e.currentTarget.value))}
                        />
                        <span>{fromYear()}</span>
                    </div>

                    <div class="dc-control">
                        <label>to</label>
                        <input type="range"
                            min={yearBounds()[0]}
                            max={yearBounds()[1]}
                            value={toYear()}
                            onInput={e => setToYear(Number(e.currentTarget.value))}
                        />
                        <span>{toYear()}</span>
                    </div>

                    <div class="dc-control">
                        <label>top N</label>
                        <input type="range"
                            min={3}
                            max={MAX_TOP_N}
                            value={topN()}
                            onInput={e => setTopN(Number(e.currentTarget.value))}
                        />
                    </div>

                    <div class="dc-control">
                        <label>window ±</label>
                        <input type="range"
                            min={0}
                            max={4}
                            value={smoothWindow()}
                            onInput={e => setSmoothWindow(Number(e.currentTarget.value))}
                        />
                    </div>

                    <div class="dc-control">
                        <label>rank by</label>
                        <select value={sortKey()} onChange={e => setSortKey(e.currentTarget.value as SortKey)}>
                            <option value="freq">frequency</option>
                            <option value="score">cosine score</option>
                        </select>
                    </div>
                </header>

                <div class="dc-scroll">
                    <svg width={svgWidth()} height={svgHeight()}>

                        <For each={years()}>
                            {(yr, i) => (
                                <text
                                    x={colX(i())}
                                    y={HEADER_H - 10}
                                    text-anchor="middle"
                                    font-size="11"
                                >
                                    {yearLabel(yr, smoothWindow())}
                                </text>
                            )}
                        </For>

                        <For each={links()}>
                            {lk => (
                                <path
                                    d={linkPath(lk.x1, lk.y1, lk.x2, lk.y2)}
                                    fill="none"
                                    stroke="#4aa59c"
                                    stroke-width="1.2"
                                    stroke-opacity={C_LINK_ALPHA}
                                />
                            )}
                        </For>

                        <For each={years()}>
                            {(yr, ci) => (
                                <For each={displaySlices().get(yr) ?? []}>
                                    {rt => {
                                        const key = `${ yr }:${ rt.token }`;
                                        const status = () => cellStatus().get(key) ?? "continuation";
                                        const color = () => statusColor(status());

                                        const x = () => colX(ci()) - CELL_WIDTH / 2;
                                        const y = () => HEADER_H + rt.rank * ROW_HEIGHT;

                                        return (
                                            <g onClick={() => setFocusToken(rt.token)}>
                                                <rect
                                                    x={x()}
                                                    y={y()}
                                                    width={CELL_WIDTH}
                                                    height={CELL_H}
                                                    fill={color()}
                                                    fill-opacity={0.12}
                                                />

                                                <text
                                                    x={x() + LABEL_PAD}
                                                    y={y() + CELL_H / 2}
                                                    dominant-baseline="middle"
                                                    font-size="10"
                                                    fill={color()}
                                                >
                                                    {rt.token}
                                                </text>
                                            </g>
                                        );
                                    }}
                                </For>
                            )}
                        </For>

                    </svg>
                </div>
            </article>
        </>
    );
};

export default DiachronicChart;