/**
 * DiachronicChart.tsx
 *
 * Heuser-style diachronic chart of contextual neighbours.
 *
 * -----------------------------------------------------------------------------
 * LAYOUT
 * -----------------------------------------------------------------------------
 *  One column per year in the filtered range.
 *  Within each column, the top-N neighbour tokens for the focal concept are
 *  ranked by occurrence frequency (or mean cosine score) and drawn as labelled
 *  cells.
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
 *      ▼
 *  ConceptEvent[]
 *      │  buildYearSlices()
 *      ▼
 *  Map<year, RankedToken[]>   — top-N per year, rank-ordered
 *      │  (reactive memos)
 *      ▼
 *  SVG — link layer + cell layer
 */

import {
    createSignal,
    createMemo,
    For,
    Show,
    type Component,
} from "solid-js";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

// -----------------------------------------------------------------------------
// Types (shared shape with the rest of the codebase)
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Derived types
// -----------------------------------------------------------------------------

interface RankedToken {
    token: string;
    rank: number;        // 0-based
    freq: number;        // occurrence count (raw, not normalised)
    meanScore: number;
    eventCount: number;  // number of distinct events in this year slice
}

type YearSlices = Map<number, RankedToken[]>;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------


const COL_WIDTH = 96;   // px per year column
const ROW_HEIGHT = 22;   // px per rank row
const LABEL_PAD = 8;    // px inside cell for text
const CELL_H = ROW_HEIGHT - 3;
const HEADER_H = 36;   // year label row height
const LEFT_MARGIN = 12;
const RIGHT_MARGIN = 12;

// Colour palette
const C_BIRTH = "#e8a838";
const C_DEATH = "#c4566a";
const C_BIRTH_DEATH = "grey";    // single-year
const C_CONTINUATION = "#4a7fa5";
const C_FOCUS = "#3ecfb2";
const C_LINK_ALPHA = 0.38;         // default link opacity
const C_LINK_FOCUS = 0.85;

// -----------------------------------------------------------------------------
// Data helpers
// -----------------------------------------------------------------------------

function scanYearRange(cd: ConceptData): [number, number] {
    let min = CORPUS_END_YEAR;
    let max = CORPUS_START_YEAR;
    for (const e of cd.events) {
        if (e.pub_year === undefined) continue;
        if (e.pub_year < min) min = e.pub_year;
        if (e.pub_year > max) max = e.pub_year;
    }
    return min <= max ? [min, max] : [CORPUS_START_YEAR, CORPUS_END_YEAR];
}

function filterByYearRange(
    events: ConceptEvent[], from: number, to: number
): ConceptEvent[] {
    return events.filter(
        e => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to
    );
}

type SortKey = "freq" | "score";

/**
 * Build per-year ranked token lists.
 *
 * @param events   Year-filtered events for the focal concept.
 * @param topN     How many tokens to retain per year.
 * @param window   Smoothing half-window: 0 = no smoothing, 1 = ±1 year, etc.
 * @param sortKey  Rank by raw occurrence frequency or mean cosine score.
 */
function buildYearSlices(
    events: ConceptEvent[],
    topN: number,
    window: number,
    sortKey: SortKey,
): YearSlices {
    if (events.length === 0) return new Map();

    // Step 1 — accumulate raw counts per (year, token).
    const raw = new Map<number, Map<string, { freq: number; scoreSum: number; eventSet: Set<string> }>>();

    for (let i = 0; i < events.length; i++) {
        const e = events[i];
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
                rec.eventSet.add(String(e.event_id ?? `idx:${ i }`));
                seenThisEvent.add(nb.token);
            }
        }
    }

    const years = [...raw.keys()].sort((a, b) => a - b);

    // Step 2 — optional smoothing: merge counts from ±window years.
    const smoothed = new Map<number, Map<string, { freq: number; scoreSum: number; eventCount: number }>>();

    for (const yr of years) {
        const merged = new Map<string, { freq: number; scoreSum: number; eventCount: number }>();

        for (let dy = -window; dy <= window; dy++) {
            const src = raw.get(yr + dy);
            if (!src) continue;
            for (const [tok, rec] of src) {
                let m = merged.get(tok);
                if (!m) { m = { freq: 0, scoreSum: 0, eventCount: 0 }; merged.set(tok, m); }
                m.freq += rec.freq;
                m.scoreSum += rec.scoreSum;
                m.eventCount += rec.eventSet.size;
            }
        }
        smoothed.set(yr, merged);
    }

    // Step 3 — rank and slice to topN.
    const slices: YearSlices = new Map();

    for (const yr of years) {
        const merged = smoothed.get(yr)!;
        const sorted = [...merged.entries()]
            .sort((a, b) => {
                const va = sortKey === "freq" ? a[1].freq : (a[1].freq > 0 ? a[1].scoreSum / a[1].freq : 0);
                const vb = sortKey === "freq" ? b[1].freq : (b[1].freq > 0 ? b[1].scoreSum / b[1].freq : 0);
                return vb - va;
            })
            .slice(0, topN);

        slices.set(yr, sorted.map(([token, rec], rank) => ({
            token,
            rank,
            freq: rec.freq,
            meanScore: rec.freq > 0 ? rec.scoreSum / rec.freq : 0,
            eventCount: rec.eventCount,
        })));
    }

    return slices;
}

// Classify a token's status in a given year.
type TokenStatus = "birth" | "death" | "birth-death" | "continuation";

function classifyStatus(
    token: string,
    year: number,
    years: number[],
    slices: YearSlices,
): TokenStatus {
    const idx = years.indexOf(year);
    const prevYear = idx > 0 ? years[idx - 1] : null;
    const nextYear = idx < years.length - 1 ? years[idx + 1] : null;

    const inPrev = prevYear !== null && slices.get(prevYear)?.some(t => t.token === token);
    const inNext = nextYear !== null && slices.get(nextYear)?.some(t => t.token === token);

    if (!inPrev && !inNext) return "birth-death";
    if (!inPrev) return "birth";
    if (nextYear && !inNext) return "death";
    return "continuation";
}

function statusColor(s: TokenStatus): string {
    if (s === "birth") return C_BIRTH;
    if (s === "death") return C_DEATH;
    if (s === "birth-death") return C_BIRTH_DEATH;
    return C_CONTINUATION;
}

// Cell Y centre for a given rank inside a column.
function cellY(rank: number): number {
    return HEADER_H + rank * ROW_HEIGHT + CELL_H / 2;
}

// X centre of a year column (0-based column index).
function colX(colIdx: number): number {
    return LEFT_MARGIN + colIdx * COL_WIDTH + COL_WIDTH / 2;
}

// Cubic bezier control points for a link between two cells.
function linkPath(
    x1: number, y1: number,
    x2: number, y2: number,
): string {
    const cx = (x1 + x2) / 2;
    return `M ${ x1 } ${ y1 } C ${ cx } ${ y1 }, ${ cx } ${ y2 }, ${ x2 } ${ y2 }`;
}

// -----------------------------------------------------------------------------
// Scoped styles
// -----------------------------------------------------------------------------

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&family=IM+Fell+English:ital@0;1&display=swap');

  .dc-root {
    display: flex;
    flex-direction: column;
    height: 100%;
    width: 100%;
    background: #0f1117;
    color: #c8cdd8;
    font-family: 'IBM Plex Mono', 'Courier New', monospace;
  }

  .dc-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: .5rem 1rem;
    background: #13161f;
    border-bottom: 1px solid #252836;
    flex-wrap: wrap;
    flex-shrink: 0;
  }

  .dc-title {
    font-family: 'IM Fell English', 'Palatino Linotype', serif;
    font-size: 1.05rem;
    font-style: italic;
    color: #8a9bb5;
    letter-spacing: .03em;
    white-space: nowrap;
  }

  .dc-control {
    display: flex;
    align-items: center;
    gap: .45rem;
    font-size: .72rem;
    color: #6a7590;
  }

  .dc-control label {
    white-space: nowrap;
  }

  .dc-control select,
  .dc-control input[type=range] {
    background: #1c2030;
    border: 1px solid #2e3348;
    color: #a0aec0;
    border-radius: 3px;
    padding: .15rem .3rem;
    font-family: inherit;
    font-size: .72rem;
  }

  .dc-control input[type=range] {
    width: 80px;
    accent-color: #3ecfb2;
    border: none;
    padding: 0;
  }

  .dc-val {
    min-width: 1.8rem;
    color: #c8cdd8;
  }

  .dc-scroll {
    flex: 1;
    overflow-x: auto;
    overflow-y: auto;
    padding: 0 0 1rem 0;
  }

  .dc-svg {
    display: block;
  }

  .dc-legend {
    display: flex;
    gap: 1.2rem;
    align-items: center;
    padding: .35rem 1rem;
    font-size: .68rem;
    color: #5a6580;
    border-top: 1px solid #1e2130;
    flex-shrink: 0;
  }

  .dc-legend-item {
    display: flex;
    align-items: center;
    gap: .35rem;
  }

  .dc-legend-swatch {
    width: 10px;
    height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }

  .dc-focus-label {
    margin-left: auto;
    color: #3ecfb2;
    font-size: .7rem;
  }
`;

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

const DiachronicChart: Component<Props> = (props) => {
    const concepts = Object.keys(props.data);

    const [concept, setConcept] = createSignal(concepts[0] ?? "");
    const [topN, setTopN] = createSignal(12);
    const [smoothWindow, setSmoothWindow] = createSignal(0);
    const [sortKey, setSortKey] = createSignal<SortKey>("freq");
    const [fromYear, setFromYear] = createSignal<number>(CORPUS_START_YEAR);
    const [toYear, setToYear] = createSignal<number>(CORPUS_END_YEAR);
    const [focusToken, setFocusToken] = createSignal<string | null>(null);

    // Year bounds for this concept
    const yearBounds = createMemo<[number, number]>(() => {
        const cd = props.data[concept()];
        if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
        return scanYearRange(cd);
    });

    // Initialise range when concept changes
    createMemo(() => {
        const [min, max] = yearBounds();
        setFromYear(min);
        setToYear(max);
    });

    // Filtered events
    const filteredEvents = createMemo(() => {
        const cd = props.data[concept()];
        if (!cd) return [];
        return filterByYearRange(cd.events, fromYear(), toYear());
    });

    // Year slices
    const slices = createMemo<YearSlices>(() =>
        buildYearSlices(filteredEvents(), topN(), smoothWindow(), sortKey())
    );

    // Ordered year list
    const years = createMemo<number[]>(() =>
        [...slices().keys()].sort((a, b) => a - b)
    );

    // SVG dimensions
    const svgWidth = createMemo(() =>
        LEFT_MARGIN + years().length * COL_WIDTH + RIGHT_MARGIN
    );
    const svgHeight = createMemo(() =>
        HEADER_H + topN() * ROW_HEIGHT + 12
    );

    // Pre-compute links: for each adjacent year pair, find shared tokens.
    interface Link {
        token: string;
        x1: number; y1: number;
        x2: number; y2: number;
        status: TokenStatus;
    }

    const links = createMemo<Link[]>(() => {
        const yrs = years();
        const sl = slices();
        const out: Link[] = [];

        for (let c = 0; c < yrs.length - 1; c++) {
            const yr = yrs[c];
            const yrN = yrs[c + 1];

            // Only draw links between *adjacent* years in the corpus (gap ≤ 1+smoothWindow)
            if (yrN - yr > 1 + smoothWindow()) continue;

            const colA = sl.get(yr) ?? [];
            const colB = sl.get(yrN) ?? [];
            const mapB = new Map(colB.map(t => [t.token, t]));

            for (const a of colA) {
                const b = mapB.get(a.token);
                if (!b) continue;
                out.push({
                    token: a.token,
                    x1: colX(c) + COL_WIDTH / 2 - 2,
                    y1: cellY(a.rank),
                    x2: colX(c + 1) - COL_WIDTH / 2 + 2,
                    y2: cellY(b.rank),
                    status: classifyStatus(a.token, yr, yrs, sl),
                });
            }
        }
        return out;
    });

    // Per-cell status cache (avoids recomputing inside For loops)
    const cellStatus = createMemo(() => {
        const yrs = years();
        const sl = slices();
        const map = new Map<string, TokenStatus>(); // key = `${year}:${token}`
        for (const yr of yrs) {
            for (const rt of sl.get(yr) ?? []) {
                map.set(`${ yr }:${ rt.token }`, classifyStatus(rt.token, yr, yrs, sl));
            }
        }
        return map;
    });

    return (
        <>
            <style>{STYLES}</style>
            <div class="dc-root">

                {/* ── Header ───────────────────────────────────────────────── */}
                <div class="dc-header">

                    <span class="dc-title">diachronic neighbours</span>

                    {/* Concept */}
                    <div class="dc-control">
                        <label>concept</label>
                        <select value={concept()} onChange={e => {
                            setConcept(e.currentTarget.value);
                            setFocusToken(null);
                        }}>
                            <For each={concepts}>{c => <option value={c}>{c}</option>}</For>
                        </select>
                    </div>

                    {/* Year range */}
                    <div class="dc-control">
                        <label>from</label>
                        <input type="range"
                            min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                            value={fromYear()}
                            onInput={e => setFromYear(Math.min(Number(e.currentTarget.value), toYear()))}
                        />
                        <span class="dc-val">{fromYear()}</span>
                    </div>

                    <div class="dc-control">
                        <label>to</label>
                        <input type="range"
                            min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                            value={toYear()}
                            onInput={e => setToYear(Math.max(Number(e.currentTarget.value), fromYear()))}
                        />
                        <span class="dc-val">{toYear()}</span>
                    </div>

                    {/* Top N */}
                    <div class="dc-control">
                        <label>top N</label>
                        <input type="range" min={3} max={25} step={1}
                            value={topN()}
                            onInput={e => setTopN(Number(e.currentTarget.value))}
                        />
                        <span class="dc-val">{topN()}</span>
                    </div>

                    {/* Smoothing */}
                    <div class="dc-control">
                        <label>window ±</label>
                        <input type="range" min={0} max={4} step={1}
                            value={smoothWindow()}
                            onInput={e => setSmoothWindow(Number(e.currentTarget.value))}
                        />
                        <span class="dc-val">{smoothWindow()}</span>
                    </div>

                    {/* Sort key */}
                    <div class="dc-control">
                        <label>rank by</label>
                        <select value={sortKey()} onChange={e => setSortKey(e.currentTarget.value as SortKey)}>
                            <option value="freq">frequency</option>
                            <option value="score">cosine score</option>
                        </select>
                    </div>

                    {/* Clear focus */}
                    <Show when={focusToken()}>
                        <button
                            style={{
                                background: "none", border: "none", cursor: "pointer",
                                color: C_FOCUS, "font-family": "inherit", "font-size": ".72rem",
                                "margin-left": "auto", display: "flex", "align-items": "center", gap: ".3rem"
                            }}
                            onClick={() => setFocusToken(null)}
                        >
                            <span style={{ "letter-spacing": ".05em" }}>✕</span>
                            <span>{focusToken()}</span>
                        </button>
                    </Show>

                </div>

                {/* ── Chart ────────────────────────────────────────────────── */}
                <div class="dc-scroll">
                    <svg
                        class="dc-svg"
                        width={svgWidth()}
                        height={svgHeight()}
                    >

                        {/* Year column headers */}
                        <For each={years()}>
                            {(yr, i) => (
                                <text
                                    x={colX(i())}
                                    y={HEADER_H - 10}
                                    text-anchor="middle"
                                    font-family="'IBM Plex Mono', monospace"
                                    font-size="11"
                                    fill={
                                        filteredEvents().filter(e => e.pub_year === yr).length > 0
                                            ? "#8090b0"
                                            : "#3a4060"
                                    }
                                >
                                    {yr}
                                </text>
                            )}
                        </For>

                        {/* Thin column separator lines */}
                        <For each={years()}>
                            {(_, i) => (
                                <line
                                    x1={colX(i()) - COL_WIDTH / 2 + 2}
                                    y1={HEADER_H - 4}
                                    x2={colX(i()) - COL_WIDTH / 2 + 2}
                                    y2={svgHeight() - 8}
                                    stroke="#1e2234"
                                    stroke-width="1"
                                />
                            )}
                        </For>

                        {/* ── Link layer (behind cells) ── */}
                        <For each={links()}>
                            {(lk) => {
                                const isFocus = () => focusToken() === lk.token;
                                const isAnyFocus = () => focusToken() !== null;
                                const col = () => isFocus() ? C_FOCUS : statusColor(lk.status);
                                const op = () => isAnyFocus()
                                    ? (isFocus() ? C_LINK_FOCUS : 0.06)
                                    : C_LINK_ALPHA;

                                return (
                                    <path
                                        d={linkPath(lk.x1, lk.y1, lk.x2, lk.y2)}
                                        fill="none"
                                        stroke={col()}
                                        stroke-width={isFocus() ? 2 : 1.2}
                                        stroke-opacity={op()}
                                        style={{ transition: "stroke-opacity 0.15s, stroke-width 0.1s" }}
                                    />
                                );
                            }}
                        </For>

                        {/* ── Cell layer ── */}
                        <For each={years()}>
                            {(yr, ci) => (
                                <For each={slices().get(yr) ?? []}>
                                    {(rt) => {
                                        const key = `${ yr }:${ rt.token }`;
                                        const status = () => cellStatus().get(key) ?? "continuation";
                                        const isFocus = () => focusToken() === rt.token;
                                        const isAnyFocus = () => focusToken() !== null;

                                        const cellColor = () => isFocus() ? C_FOCUS : statusColor(status());
                                        const textOp = () => isAnyFocus() ? (isFocus() ? 1 : 0.2) : 0.88;
                                        const rectOp = () => isAnyFocus() ? (isFocus() ? 0.22 : 0.04) : 0.13;

                                        const x = () => colX(ci()) - COL_WIDTH / 2 + 2;
                                        const y = () => HEADER_H + rt.rank * ROW_HEIGHT;
                                        const w = COL_WIDTH - 4;

                                        return (
                                            <g
                                                cursor="pointer"
                                                onClick={() => setFocusToken(prev => prev === rt.token ? null : rt.token)}
                                                style={{ transition: "opacity 0.12s" }}
                                            >
                                                {/* Background rect */}
                                                <rect
                                                    x={x()} y={y()}
                                                    width={w} height={CELL_H}
                                                    rx={2}
                                                    fill={cellColor()}
                                                    fill-opacity={rectOp()}
                                                    style={{ transition: "fill-opacity 0.12s" }}
                                                />

                                                {/* Token label */}
                                                <text
                                                    x={x() + LABEL_PAD}
                                                    y={y() + CELL_H / 2 + 1}
                                                    dominant-baseline="middle"
                                                    font-family="'IBM Plex Mono', monospace"
                                                    font-size="10.5"
                                                    font-weight={isFocus() ? "500" : "300"}
                                                    fill={cellColor()}
                                                    fill-opacity={textOp()}
                                                    style={{ transition: "fill-opacity 0.12s, font-weight 0.1s" }}
                                                >
                                                    {rt.token}
                                                </text>

                                                {/* Rank badge (right side) */}
                                                <text
                                                    x={x() + w - LABEL_PAD}
                                                    y={y() + CELL_H / 2 + 1}
                                                    dominant-baseline="middle"
                                                    text-anchor="end"
                                                    font-family="'IBM Plex Mono', monospace"
                                                    font-size="8.5"
                                                    fill={cellColor()}
                                                    fill-opacity={textOp() * 0.5}
                                                >
                                                    {rt.rank + 1}
                                                </text>

                                                {/* Invisible wider hit target */}
                                                <rect
                                                    x={x()} y={y()}
                                                    width={w} height={CELL_H}
                                                    fill="transparent"
                                                />
                                            </g>
                                        );
                                    }}
                                </For>
                            )}
                        </For>

                    </svg>
                </div>

                {/* ── Legend ───────────────────────────────────────────────── */}
                <div class="dc-legend">
                    <div class="dc-legend-item">
                        <div class="dc-legend-swatch" style={{ background: C_BIRTH }} />
                        <span>birth</span>
                    </div>
                    <div class="dc-legend-item">
                        <div class="dc-legend-swatch" style={{ background: C_DEATH }} />
                        <span>death</span>
                    </div>
                    <div class="dc-legend-item">
                        <div class="dc-legend-swatch" style={{ background: C_BIRTH_DEATH }} />
                        <span>birth + death</span>
                    </div>
                    <div class="dc-legend-item">
                        <div class="dc-legend-swatch" style={{ background: C_CONTINUATION }} />
                        <span>continuation</span>
                    </div>
                    <div class="dc-legend-item">
                        <div class="dc-legend-swatch" style={{ background: C_FOCUS }} />
                        <span>focus</span>
                    </div>
                    <span style={{ "margin-left": "auto", color: "#3a4560" }}>
                        {years().length} years · {filteredEvents().length} events · click cell to focus token
                    </span>
                </div>

            </div>
        </>
    );
};

export default DiachronicChart;