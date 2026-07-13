import {
  createSignal,
  createMemo,
  createResource,
  For,
  type Component,
  Show,
} from "solid-js";

import { type SortKey, type YearSlices, buildYearSlices } from "../../lib/yearUtils";
import { getYearFiltered } from "../../state/controls.selectors";
import ControlsHeader from "../ControlsHeader";
import { controls } from "../../state/controls.store";
import "./style.css";

const CELL_WIDTH = 92;
const COL_GAP = 32;
const COL_WIDTH = CELL_WIDTH + COL_GAP;

const ROW_HEIGHT = 22;
const LABEL_PAD = 2;
const CELL_H = ROW_HEIGHT - 3;
const HEADER_H = 36;
const LEFT_MARGIN = 12;
const RIGHT_MARGIN = 12;

export const C_BIRTH = "hsl(98, 79%, 56%)";
export const C_DEATH = "#ee7188";
export const C_BIRTH_DEATH = "#4a7fa5";
export const C_CONTINUATION = "#4aa59c";
export const C_FOCUS = "#3ecfb2";
const C_RECT_UNFOCUS = 0.05;
const C_RECT_FOCUS = 0.18;
const C_LINK_ALPHA = 0.6;
const C_LINK_FOCUS = 0.85;
const C_LINK_UNFOCUS = 0.05;
const C_LINK_UNFOCUS_TEXT = 0.9;

export type TokenStatus = "birth" | "death" | "birth-death" | "continuation";

export function classifyStatus(
  token: string,
  year: number,
  years: number[],
  slices: YearSlices,
): TokenStatus {
  const idx = years.indexOf(year);
  const previousYears = years.slice(0, idx);
  const futureYears = years.slice(idx + 1);

  const existedBefore = previousYears.some((y) => slices.get(y)?.some((t) => t.token === token),);

  const existsLater = futureYears.some((y) => slices.get(y)?.some((t) => t.token === token),);

  const presentThisYear = slices.get(year)?.some((t) => t.token === token) ?? false;

  if (!presentThisYear) return "continuation";
  if (!existedBefore && !existsLater) return "birth-death";
  if (!existedBefore) return "birth";
  if (!existsLater) return "death";
  return "continuation";
}

function yearLabel(year: number, window: number): string {
  if (window === 0) return String(year);
  return `${ year - window }-${ year + window }`;
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

type GridPosition = {
  col: number;
  rank: number;
}

const DiachronicChart: Component = () => {
  let scrollRef: HTMLDivElement | undefined;

  const [smoothWindow, setSmoothWindow] = createSignal(0);
  const [sortKey, setSortKey] = createSignal<SortKey>("freq");
  const [focusPos, setFocusPos] = createSignal<GridPosition | null>(null);

  const [filteredEventsResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredEvents = () => filteredEventsResource() ?? [];

  const displaySlices = createMemo<YearSlices>(() =>
    buildYearSlices(filteredEvents(), controls.topN, smoothWindow(), sortKey()),
  );

  const rawSlices = createMemo<YearSlices>(() =>
    buildYearSlices(filteredEvents(), controls.topN, 0, sortKey()),
  );

  const years = createMemo<number[]>(() =>
    [...displaySlices().keys()].sort((a, b) => a - b),
  );

  const focusToken = createMemo<string | null>(() => {
    const pos = focusPos();
    if (!pos) return null;
    const yr = years()[pos.col];
    if (yr === undefined) return null;
    const col = displaySlices().get(yr) ?? [];
    return col.find(t => t.rank === pos.rank)?.token ?? null;
  });

  const svgWidth = createMemo(
    () => LEFT_MARGIN + years().length * COL_WIDTH + RIGHT_MARGIN,
  );

  const svgHeight = createMemo(
    () => HEADER_H + controls.topN * ROW_HEIGHT + 12,
  );

  function swapTokenForLinked(pos: GridPosition, reverseDirection: boolean, jumpToEnd: boolean) {
    const token = focusToken();
    if (!token) return;

    const yrs = years();
    const sl = displaySlices();
    const { col } = pos;

    const occurrences = yrs
      .map((yr, i) => ({ i, items: sl.get(yr) ?? [] }))
      .filter(({ items }) => items.some(t => t.token === token))
      .map(({ i }) => i);

    if (occurrences.length < 2) return;

    let nextCol, nextRank;

    if (jumpToEnd) {
      nextCol = reverseDirection
        ? occurrences[occurrences.length - 1]
        : nextCol = occurrences.reverse().find(i => i > col) ?? occurrences[0]
    }
    else {
      nextCol = reverseDirection
        ? occurrences.reverse().find(i => i < col) ?? occurrences[occurrences.length - 1]
        : occurrences.find(i => i > col) ?? occurrences[0];
    }

    nextRank = (sl.get(yrs[nextCol]) ?? []).find(t => t.token === token)!.rank;

    setFocusPos({ col: nextCol, rank: nextRank });
    scrollColIntoView(nextCol);
    return;
  }


  function scrollColIntoView(col: number) {
    const scrollEl = scrollRef;
    if (!scrollEl) return;

    const x = colX(col) - CELL_WIDTH / 2;
    const cellRight = x + CELL_WIDTH;
    const scrollLeft = scrollEl.scrollLeft;
    const clientWidth = scrollEl.clientWidth;

    if (x < scrollLeft || cellRight > scrollLeft + clientWidth) {
      if (x < scrollLeft) {
        scrollEl.scrollLeft = x - LEFT_MARGIN;
      } else {
        scrollEl.scrollLeft = cellRight - clientWidth + RIGHT_MARGIN;
      }
    }
  }


  function handleKeyDown(e: KeyboardEvent) {
    const yrs = years();
    const sl = displaySlices();
    const pos = focusPos();

    if (!["Enter", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Escape"].includes(e.key)) return;
    e.preventDefault();

    if (e.key === "Escape") {
      setFocusPos(null);
      return;
    }

    if (!pos) {
      setFocusPos({ col: 0, rank: 0 });
      scrollColIntoView(0);
      return;
    }

    if (e.key === "Enter") {
      return swapTokenForLinked(
        pos,
        e.shiftKey,
        e.altKey || e.ctrlKey || e.metaKey
      );
    }

    const { col, rank } = pos;
    let [newCol, newRank] = [col, rank];
    const token = focusToken();

    if (e.key === "ArrowUp") {
      newRank = rank - 1;
    }
    else if (e.key === "ArrowDown") {
      const maxRank = (sl.get(yrs[col]) ?? []).length - 1;
      newRank = Math.min(maxRank, rank + 1);
    }
    else {
      newCol = e.key === "ArrowLeft"
        ? Math.max(0, col - 1)
        : Math.min(yrs.length - 1, col + 1);

      if (newCol === col) return;

      const newColItems = sl.get(yrs[newCol]) ?? [];
      const sameToken = token ? newColItems.find(t => t.token === token) : undefined;
      newRank = sameToken
        ? sameToken.rank
        : Math.min(rank, Math.max(0, newColItems.length - 1));
    }

    setFocusPos({ col: newCol, rank: newRank });
    scrollColIntoView(newCol);
  }


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

      const mapB = new Map(colB.map((t) => [t.token, t]));

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
          if (col.some((t) => t.token === token)) positions.push(i);
        }

        for (let i = 0; i < positions.length - 1; i++) {
          const aIdx = positions[i];
          const bIdx = positions[i + 1];

          if (bIdx === aIdx + 1) continue;

          const yrA = yrs[aIdx];
          const yrB = yrs[bIdx];

          const colA = sl.get(yrA) ?? [];
          const colB = sl.get(yrB) ?? [];

          const a = colA.find((t) => t.token === token);
          const b = colB.find((t) => t.token === token);

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
        map.set(`${ yr }:${ rt.token }`, classifyStatus(rt.token, yr, yrs, sl));
      }
    }

    return map;
  });

  return (
    <article class="background no-padding no-margin">
      <ControlsHeader title="Diachronic Neighbours" topN={true}>
        <hr class="divider vertical max no-margin no-padding" />

        <div class="field border middle-align small">
          <select
            style="max-width: 7rem"
            value={sortKey()}
            onChange={(e) => setSortKey(e.currentTarget.value as SortKey)}
          >
            <option value="freq">Frequency</option>
            <option value="score">Cosine</option>
          </select>
          <span class="tooltip bottom">Rank by</span>
        </div>

        <hr class="divider vertical max no-margin no-padding" />

        <div class="field middle-align" style="max-width: 4rem">
          <div class="slider">
            <input
              type="range"
              min={0}
              max={4}
              value={smoothWindow()}
              onInput={(e) => setSmoothWindow(Number(e.currentTarget.value))}
            />
            <span class="tooltip bottom" />
          </div>
          <span class="tooltip bottom">Smoothing</span>
        </div>

        <div class="field middle-align fill">
          <h2 class="large extra-text large-opacity secondary-text">
            {focusToken()}
          </h2>
        </div>
      </ControlsHeader>

      <Show when={filteredEventsResource.loading}>
        <progress />
      </Show>

      <div class="scroll" ref={scrollRef}>
        <svg
          width={svgWidth()}
          height={svgHeight()}
          tabIndex={0}
          onKeyDown={handleKeyDown}
          style="outline: none;"
        >
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
            {(lk) => (
              <path
                d={linkPath(lk.x1, lk.y1, lk.x2, lk.y2)}
                fill="none"
                stroke="#4aa59c"
                stroke-width="3"
                stroke-opacity={
                  focusToken()
                    ? lk.token === focusToken()
                      ? C_LINK_FOCUS
                      : C_LINK_UNFOCUS
                    : C_LINK_ALPHA
                }
              />
            )}
          </For>

          <For each={years()}>
            {(yr, ci) => (
              <For each={displaySlices().get(yr) ?? []}>
                {(rt) => {
                  const key = `${ yr }:${ rt.token }`;
                  const status = () => cellStatus().get(key) ?? "continuation";
                  const color = () => statusColor(status());
                  const isFocused = () =>
                    !focusToken() || focusToken() === rt.token;
                  const isCursor = () => {
                    const pos = focusPos();
                    return pos?.col === ci() && pos?.rank === rt.rank;
                  };
                  const x = () => colX(ci()) - CELL_WIDTH / 2;
                  const y = () => HEADER_H + rt.rank * ROW_HEIGHT;

                  return (
                    <g
                      onClick={() =>
                        setFocusPos(prev =>
                          prev?.col === ci() && prev?.rank === rt.rank
                            ? null
                            : { col: ci(), rank: rt.rank }
                        )
                      }
                    >
                      <rect
                        x={x()}
                        y={y()}
                        width={CELL_WIDTH}
                        height={CELL_H}
                        fill={color()}
                        stroke={isCursor() ? C_FOCUS : "none"}
                        stroke-width={isCursor() ? 1.2 : 0}
                        fill-opacity={isFocused() ? C_RECT_FOCUS : C_RECT_UNFOCUS}
                      />
                      <text
                        x={x() + LABEL_PAD}
                        y={y() + CELL_H / 2}
                        dominant-baseline="middle"
                        font-size="14"
                        font-family="Junicode"
                        fill={color()}
                        opacity={isFocused() ? 1 : C_LINK_UNFOCUS_TEXT}
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
  );
};

export default DiachronicChart;