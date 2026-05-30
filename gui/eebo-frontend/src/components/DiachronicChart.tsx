import {
    createSignal,
    createMemo,
    For,
    type Component,
} from "solid-js";

import { filterByYearRange, } from "../lib/contextGraphUtils";
import { tier2Data } from "../state/tier2data.store";
import ControlsHeader from "./ControlsHeader";
import { controls } from "../state/controls.store";
import type { ConceptEvent } from "../types/context-graph.types";

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
const C_DEATH = "#ee7188";
const C_BIRTH_DEATH = "#4a7fa5";
const C_CONTINUATION = "#4aa59c";
const C_FOCUS = "#3ecfb2";
const C_LINK_ALPHA = 0.38;
const C_LINK_FOCUS = 0.85;

function yearLabel(year: number, window: number): string {
    if (window === 0) return String(year);
    return `${ year - window }–${ year + window }`;
}

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

    const existedBefore = previousYears.some(y => {
        const col = slices.get(y);
        return col?.some(t => t.token === token);
    });

    const existsLater = futureYears.some(y => {
        const col = slices.get(y);
        return col?.some(t => t.token === token);
    });

    const presentThisYear =
        slices.get(year)?.some(t => t.token === token) ?? false;

    if (!presentThisYear) return "continuation";

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

function linkPath(x1: number, y1: number, x2: number, y2: number): string {
    const cx = (x1 + x2) / 2;
    return `M ${ x1 } ${ y1 } C ${ cx } ${ y1 }, ${ cx } ${ y2 }, ${ x2 } ${ y2 }`;
}

const DiachronicChart: Component = () => {
    const [smoothWindow, setSmoothWindow] = createSignal(0);
    const [sortKey, setSortKey] = createSignal<SortKey>("freq");
    const [focusToken, setFocusToken] = createSignal<string | null>(null);

    const filteredEvents = createMemo(() => {
        const cd = tier2Data[controls.concept];
        if (!cd) return [];
        return filterByYearRange(cd.events, controls.fromYear, controls.toYear);
    });

    const displaySlices = createMemo<YearSlices>(() =>
        buildYearSlices(
            filteredEvents(),
            controls.topN,
            smoothWindow(),
            sortKey()
        )
    );

    const rawSlices = createMemo<YearSlices>(() =>
        buildYearSlices(
            filteredEvents(),
            controls.topN,
            0,
            sortKey()
        )
    );

    const years = createMemo<number[]>(() =>
        [...displaySlices().keys()].sort((a, b) => a - b)
    );

    const svgWidth = createMemo(() =>
        LEFT_MARGIN + years().length * COL_WIDTH + RIGHT_MARGIN
    );

    const svgHeight = createMemo(() =>
        HEADER_H + controls.topN * ROW_HEIGHT + 12
    );

    const links = createMemo(() => {
        const focus = focusToken();
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
                if (focus && a.token !== focus) continue;

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

            if (focus) {
                const token = focus;
                const positions: number[] = [];

                for (let i = 0; i < yrs.length; i++) {
                    const yrX = yrs[i];
                    const col = sl.get(yrX) ?? [];
                    if (col.some(t => t.token === token)) {
                        positions.push(i);
                    }
                }

                for (let i = 0; i < positions.length - 1; i++) {
                    const aIdx = positions[i];
                    const bIdx = positions[i + 1];

                    if (bIdx === aIdx + 1) continue;

                    const yrA = yrs[aIdx];
                    const yrB = yrs[bIdx];

                    const colA = sl.get(yrA) ?? [];
                    const colB = sl.get(yrB) ?? [];

                    const a = colA.find(t => t.token === token);
                    const b = colB.find(t => t.token === token);

                    if (!a || !b) continue;

                    out.push({
                        token,
                        x1: colX(aIdx) + CELL_WIDTH / 2,
                        y1: cellY(a.rank),
                        x2: colX(bIdx) - CELL_WIDTH / 2,
                        y2: cellY(b.rank),
                    });
                }
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
        <article class="background  no-padding no-margin">
            <ControlsHeader title="Diachronic Neighbours" fdgControls={false}>

                <hr class="divider vertical max no-margin no-padding" />

                <div class="field middle-align border">
                    <select value={sortKey()} onChange={e => setSortKey(e.currentTarget.value as SortKey)}>
                        <option value="freq">frequency</option>
                        <option value="score">cosine score</option>
                    </select>
                    <output>Rank by</output>
                </div>

                <hr class="divider vertical max no-margin no-padding" />

                <div class="field middle-align">
                    <div class="slider tiny">
                        <input
                            type="range"
                            min={0}
                            max={4}
                            value={smoothWindow()}
                            onInput={e => setSmoothWindow(Number(e.currentTarget.value))}
                        />
                        <span class="tooltip bottom" />
                    </div>
                    <output class="small-padding top-padding">
                        Smoothing
                    </output>
                </div>
            </ControlsHeader>

            <div class="scroll">

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
                                stroke-opacity={
                                    focusToken()
                                        ? (lk.token === focusToken() ? C_LINK_FOCUS : 0.08)
                                        : C_LINK_ALPHA
                                }
                            />
                        )}
                    </For>

                    <For each={years()}>
                        {(yr, ci) => (
                            <For each={displaySlices().get(yr) ?? []}>
                                {rt => {
                                    const key = `${ yr }:${ rt.token }`;
                                    const status = () =>
                                        cellStatus().get(key) ?? "continuation";
                                    const color = () => statusColor(status());

                                    const isFocused = () =>
                                        !focusToken() || focusToken() === rt.token;

                                    const x = () => colX(ci()) - CELL_WIDTH / 2;
                                    const y = () => HEADER_H + rt.rank * ROW_HEIGHT;

                                    return (
                                        <g
                                            onClick={() =>
                                                setFocusToken(prev =>
                                                    prev === rt.token ? null : rt.token
                                                )
                                            }
                                        >
                                            <rect
                                                x={x()}
                                                y={y()}
                                                width={CELL_WIDTH}
                                                height={CELL_H}
                                                fill={color()}
                                                stroke={focusToken() === rt.token ? C_FOCUS : "none"}
                                                stroke-width={focusToken() === rt.token ? 1.2 : 0}
                                                fill-opacity={isFocused() ? 0.18 : 0.04}
                                            />

                                            <text
                                                x={x() + LABEL_PAD}
                                                y={y() + CELL_H / 2}
                                                dominant-baseline="middle"
                                                font-size="10"
                                                fill={color()}
                                                opacity={isFocused() ? 1 : 0.25}
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
        </article >
    );
};

export default DiachronicChart;