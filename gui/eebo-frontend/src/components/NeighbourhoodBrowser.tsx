/**
 * NeighbourhoodBrowser.tsx
 *
 * Faceted browser for FAISS-derived KNN neighbourhood data.
 *
 * PURPOSE
 * Surfaces the contextual evidence embedded in Tier 2 KNN data: for a given
 * concept word, what semantic neighbours appear alongside it across the corpus,
 * and in which documents and years?
 *
 *
 * STATE MODEL
 *  concept          : which concept word to browse
 *  fromYear/toYear  : temporal filter applied to events
 *  yearMode         : single | range
 *  selectedEventId  : event whose neighbour list fills the centre panel
 *  focusToken       : neighbour token highlighted globally; when set:
 *                       - left panel dims events that lack it
 *                       - right panel shows all docs carrying it (not just
 *                         the selected event's doc)
 *                       - centre panel highlights it in the chip list
 *
 * DATA FLOW
 *   tier2Data (Tier2Data)
 *       │  filterByYearRange()
 *       |
 *   ConceptEvent[]          — year-filtered events
 *       │  buildNeighbourIndex()
 *       |
 *   NeighbourIndex          — token : {events, meanScore, docSet}
 *       │  (reactive memos, no further transforms)
 *       |
 *   Left / Centre / Right panels
 *
 * No graph construction, no force simulation, no D3.
 */

import {
  createSignal,
  createMemo,
  For,
  Show,
  type Component,
  createEffect,
} from "solid-js";
import Sparkline from "./NeighbourhoodBrowser/SparkLine";
import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";
import { tier2Data } from "../state/tier2data.store";
import type { ConceptData, Neighbour } from "../types/context-graph.types";
import type { ConceptEvent } from "./CosmosContextGraph/types";
import { createTokenWindowResource } from "../services/tokenWindowApi";


/**
 * Aggregated view of a neighbour token across all events in the current
 * filtered set.  Built once per filter change by buildNeighbourIndex().
 */
interface NeighbourSummary {
  token: string;
  /** raw neighbour occurrences across corpus slice */
  occurrenceCount: number;
  /** Number of distinct events in which this token appears as a neighbour. */
  eventCount: number;
  /** Mean cosine score across all occurrences. */
  meanScore: number;
  /** Set of doc_ids from events carrying this token. */
  docIds: Set<string>;
  /**
   * doc_id : pub_year pairs from events carrying this token.
   * Stored directly during index construction to avoid re-scanning
   * yearFiltered() in rightPanelDocs.
   */
  docYears: Map<string, number | undefined>;
  /** event_id keys (or idx:N fallbacks) of events carrying this token. */
  eventKeys: Set<string>;
}

/** Map from token string : NeighbourSummary. */
type NeighbourIndex = Map<string, NeighbourSummary>;


function scanYearRange(cd: ConceptData): [number, number] {
  let min = CORPUS_END_YEAR;
  let max = CORPUS_START_YEAR;
  for (const e of cd.events) {
    if (e.pub_year === undefined) continue;
    if (e.pub_year < min) min = e.pub_year;
    if (e.pub_year > max) max = e.pub_year;
  }
  if (min > max) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
  return [min, max];
}

function filterByYearRange(
  events: ConceptEvent[],
  from: number,
  to: number,
): ConceptEvent[] {
  return events.filter(
    (e) => e.pub_year !== undefined && e.pub_year >= from && e.pub_year <= to
  );
}

function eventKey(event: ConceptEvent, idx: number): string {
  return event.event_id !== undefined ? String(event.event_id) : `idx:${ idx }`;
}

function buildNeighbourIndex(events: ConceptEvent[]): NeighbourIndex {
  const index: NeighbourIndex = new Map();

  for (let idx = 0; idx < events.length; idx++) {
    const event = events[idx];
    const key = eventKey(event, idx);

    // track per-event deduplication of tokens
    const seenInEvent = new Set<string>();

    for (const nb of event.neighbours) {
      let summary = index.get(nb.token);

      if (!summary) {
        summary = {
          token: nb.token,
          occurrenceCount: 0,
          eventCount: 0,
          meanScore: 0,
          docIds: new Set(),
          docYears: new Map(),
          eventKeys: new Set(),
        };
        index.set(nb.token, summary);
      }

      summary.occurrenceCount += 1;

      // distributional signal (once per event)
      if (!seenInEvent.has(nb.token)) {
        summary.eventCount += 1;
        summary.eventKeys.add(key);
        seenInEvent.add(nb.token);
      }

      // score accumulation still occurrence-weighted
      summary.meanScore += nb.score;

      if (event.doc_id) {
        summary.docIds.add(event.doc_id);
        if (!summary.docYears.has(event.doc_id)) {
          summary.docYears.set(event.doc_id, event.pub_year);
        }
      }
    }
  }

  for (const s of index.values()) {
    if (s.occurrenceCount > 0) {
      s.meanScore /= s.occurrenceCount;
    }
  }

  return index;
}


const showDocument = (docId: string) =>
  window.open(`/api/doc/${ docId }`, "_blank", "noopener,noreferrer");

/** Convert a score in [scoreMin, scoreMax] to an opacity in [minOp, maxOp]. */
function scoreToOpacity(
  score: number,
  scoreMin: number,
  scoreMax: number,
  minOp = 0.5,
  maxOp = 1.0
): number {
  const range = scoreMax - scoreMin;
  if (range < 1e-9) return maxOp;
  return minOp + ((score - scoreMin) / range) * (maxOp - minOp);
}


const NeighbourhoodBrowser: Component = () => {
  const concepts = Object.keys(tier2Data);

  const [concept, setConcept] = createSignal(concepts[0] ?? "");
  const [fromYear, setFromYear] = createSignal<number>(-1);
  const [toYear, setToYear] = createSignal<number>(-1);
  const [yearMode, setYearMode] = createSignal<"single" | "range">("single");
  const [selectedEventId, setSelectedEventId] = createSignal<string | null>(null);
  const [focusToken, setFocusToken] = createSignal<string | null>(null);

  const yearBounds = createMemo<[number, number]>(() => {
    const cd = tier2Data[concept()];
    if (!cd) return [CORPUS_START_YEAR, CORPUS_END_YEAR];
    return scanYearRange(cd);
  });

  // Reset sliders when concept or mode changes - move to the onChange event to avoid reactivty loop
  createEffect(() => {
    const [min, max] = yearBounds();
    if (yearMode() === "single") {
      const mid = Math.floor((min + max) / 2);
      setFromYear(mid);
      setToYear(mid);
    } else {
      setFromYear(min);
      setToYear(max);
    }
  });

  const yearFiltered = createMemo<ConceptEvent[]>(() => {
    const cd = tier2Data[concept()];
    if (!cd) return [];
    const [min, max] = yearBounds();
    if (fromYear() <= min && toYear() >= max) return cd.events;
    return filterByYearRange(cd.events, fromYear(), toYear());
  });

  const neighbourIndex = createMemo<NeighbourIndex>(() =>
    buildNeighbourIndex(yearFiltered())
  );

  // Neighbour summaries sorted by (eventCount desc, meanScore desc)
  const sortedGlobalNeighbours = createMemo<NeighbourSummary[]>(() =>
    [...neighbourIndex().values()]
      .sort((a, b) =>
        b.eventCount - a.eventCount || b.meanScore - a.meanScore
      )
  );


  const selectedEvent = createMemo<{ event: ConceptEvent; key: string } | null>(() => {
    const id = selectedEventId();
    if (!id) return null;
    const events = yearFiltered();
    for (let idx = 0; idx < events.length; idx++) {
      const k = eventKey(events[idx], idx);
      if (k === id) return { event: events[idx], key: k };
    }
    return null;
  });

  const [windowText] = createTokenWindowResource(
    () => selectedEvent()?.event ?? null
  );

  const selectedEventNeighbours = createMemo<Neighbour[]>(() => {
    const sel = selectedEvent();
    if (!sel) return [];
    return [...sel.event.neighbours].sort((a, b) => b.score - a.score);
  });

  const selectedScoreRange = createMemo<[number, number]>(() => {
    const nbs = selectedEventNeighbours();
    if (nbs.length === 0) return [0, 1];
    let min = nbs[0].score;
    let max = nbs[0].score;
    for (let i = 1; i < nbs.length; i++) {
      if (nbs[i].score < min) min = nbs[i].score;
      if (nbs[i].score > max) max = nbs[i].score;
    }
    return [min, max];
  });

  // Document panel

  /**
   * Documents to show in the right panel.
   * - If a focusToken is set: all docs from events that carry that token,
   *   annotated with year, sorted by year.
   * - Else if an event is selected: just that event's doc_id.
   * - Else: empty.
   *
   */
  const rightPanelDocs = createMemo<Array<{ docId: string; year?: number }>>(() => {
    const ft = focusToken();
    if (ft) {
      const summary = neighbourIndex().get(ft);
      if (!summary) return [];
      return [...summary.docYears.entries()]
        .map(([docId, year]) => ({ docId, year }))
        .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
    }

    const sel = selectedEvent();
    if (sel?.event.doc_id) {
      return [{ docId: sel.event.doc_id, year: sel.event.pub_year }];
    }

    return [];
  });

  // Highlight: events that carry the focusToken

  const focusEventKeys = createMemo<Set<string>>(() => {
    const ft = focusToken();
    if (!ft) return new Set();
    return neighbourIndex().get(ft)?.eventKeys ?? new Set();
  });


  // Temporal profile

  const tokenTemporalProfile = createMemo(() => {
    const events = yearFiltered();
    const map = new Map<string, Map<number, Set<string>>>();

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
  });

  type TemporalProfile = Map<number, Set<string>>;

  function toSeries(
    profile: Map<string, TemporalProfile>,
    token: string
  ): { year: number; value: number }[] {
    const byYear = profile.get(token);
    if (!byYear) return [];

    return [...byYear.entries()]
      .map(([year, set]) => ({
        year,
        value: set.size,
      }))
      .sort((a, b) => a.year - b.year);
  }

  // UI

  return (
    <div style={{ display: "flex", "flex-direction": "column", height: "100%", width: "100%" }}>

      {/* Header */}
      <header class="center-align max surface-container-low small-padding top-padding">
        <nav>

          {/* Concept */}
          <div class="field suffix border middle-align">
            <select
              value={concept()}
              onChange={(e) => { setConcept(e.currentTarget.value); setSelectedEventId(null); setFocusToken(null); }}
            >
              <For each={concepts}>{(c) => <option value={c}>{c}</option>}</For>
            </select>
            <output>Concept</output>
          </div>

          {/* Year mode */}
          <div class="field suffix border middle-align">
            <select
              value={yearMode()}
              onChange={(e) => setYearMode(e.currentTarget.value as "single" | "range")}
            >
              <option value="single">Single year</option>
              <option value="range">Year range</option>
            </select>
            <output>Year mode</output>
          </div>

          {/* Single-year slider */}
          <Show when={yearMode() === "single"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range" min={CORPUS_START_YEAR} max={CORPUS_END_YEAR} step={1}
                  value={fromYear()}
                  onInput={(e) => { const v = Number(e.currentTarget.value); setFromYear(v); setToYear(v); }}
                />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                {fromYear()} ({yearFiltered().length} events)
              </output>
            </div>
          </Show>

          {/* Year-range sliders */}
          <Show when={yearMode() === "range"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                  value={fromYear()}
                  onInput={(e) => setFromYear(Math.min(Number(e.currentTarget.value), toYear()))}
                />
                <input
                  type="range" min={yearBounds()[0]} max={yearBounds()[1]} step={1}
                  value={toYear()}
                  onInput={(e) => setToYear(Math.max(Number(e.currentTarget.value), fromYear()))}
                />
                <span />
                <span class="tooltip bottom" />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                <span>{fromYear()}–{toYear()}</span>
                <span class="left-padding">
                  {yearFiltered().length}/{tier2Data[concept()]?.n_events ?? 0} events
                </span>
              </output>
            </div>
          </Show>

        </nav>
      </header>

      {/* Three-column main area */}
      <div class="grid background no-margin" style={{ display: "flex", flex: "1", overflow: "hidden" }} >

        {/* LEFT: event list */}
        <nav class="s3 surface-container"
          style={{ "flex-shrink": "0", "overflow-y": "auto", display: "flex", "flex-direction": "column" }}
        >
          <div class="padding small-text bold" >
            Events
            <span class="right-align small-text left-padding medium-opacity">
              {yearFiltered().length}
            </span>
          </div>

          <For each={yearFiltered()}>
            {(event, idx) => {
              const key = () => eventKey(event, idx());
              const hasFocus = () => {
                const ft = focusToken();
                return !ft || focusEventKeys().has(key());
              };
              const isSelected = () => selectedEventId() === key();

              return (
                <button
                  class={`chip tiny-padding left-padding right-padding no-round no-margin ${ isSelected() ? "primary" : "transparent" }`}
                  style={{ opacity: hasFocus() ? 1 : 0.35, transition: "opacity 0.15s" }}
                  onClick={() => setSelectedEventId((prev) => prev === key() ? null : key())}
                >
                  <span class="tooltip top">
                    eid:{event.event_id}
                  </span>
                  <Show when={event.pub_year !== undefined} fallback={"&mbash;"}>
                    <span class="small-text">{event.pub_year}</span>
                  </Show>
                  <span class="code">
                    {event.doc_id ?? key()}
                  </span>
                  <span class="small-text medium-opacity">
                    {event.neighbours.length} neighbours
                  </span>
                </button>
              );
            }}
          </For>
        </nav>

        {/* CENTRE: neighbour tokens */}
        <section
          class="s6 surface-container"
          style={{ flex: "1", "overflow-y": "auto", display: "flex", "flex-direction": "column" }}
        >

          {/* event window token text */}
          <Show when={selectedEvent()}>
            <aside class="center-align small-padding border small-round">
              <Show when={windowText()} fallback={<div><p>Loading context…</p><progress /></div>}>
                {(windowText) => (
                  <blockquote innerHTML={windowText()}></blockquote>
                )}
              </Show>
            </aside>
          </Show>

          <div class="padding small-text bold" style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}>
            <Show when={selectedEvent()} fallback={
              <span>
                Neighbours: all events
                <span class="small-text left-padding medium-opacity">
                  {sortedGlobalNeighbours().length} tokens
                </span>
              </span>
            }>
              {(sel) => (
                <span>
                  Neighbours: {sel().event.doc_id ?? sel().key}
                  <Show when={sel().event.pub_year !== undefined}>
                    <span class="small-text left-padding medium-opacity">{sel().event.pub_year}</span>
                  </Show>
                  <span class="small-text left-padding medium-opacity">
                    {selectedEventNeighbours().length} tokens
                  </span>
                </span>
              )}
            </Show>
          </div>

          {/* Event selected: show its neighbour list as rows with score bar */}
          <Show when={selectedEvent()}>
            <div style={{ padding: "0.5rem 0" }}>
              <For each={selectedEventNeighbours()}>
                {(nb) => {
                  const [sMin, sMax] = selectedScoreRange();
                  const op = scoreToOpacity(nb.score, sMin, sMax);
                  const barPct = sMax > sMin
                    ? ((nb.score - sMin) / (sMax - sMin)) * 100
                    : 100;
                  const isFocus = () => focusToken() === nb.token;

                  return (
                    <button class="responsive max no-round left-padding right-padding"
                      style={{
                        color: 'oldlace',
                        display: "flex",
                        "align-items": "center",
                        gap: "0.5rem",
                        background: isFocus() ? "rgba(100,180,255,0.12)" : "transparent",
                        opacity: op,
                        transition: "background 0.1s",
                      }}
                      onClick={() => setFocusToken((prev) => prev === nb.token ? null : nb.token)}
                    >
                      {/* Score bar */}
                      <div style={{ width: "20%", "flex-shrink": "0", position: "relative", height: "6px", background: "rgba(255,255,255,0.08)", "border-radius": "3px" }}>
                        <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: `${ barPct }%`, background: "rgba(100,180,255,0.7)", "border-radius": "3px" }} />
                      </div>
                      {/* Token */}
                      <span style={{ "font-family": "'IBM Plex Mono', monospace", "font-size": "0.85rem", flex: "1", "text-align": "left" }}>
                        {nb.token}
                      </span>
                      {/* Score */}
                      <span class="small-text" style={{ opacity: 0.55, "flex-shrink": "0" }}>
                        {nb.score.toFixed(4)}
                      </span>
                      {/* Year from neighbour if present */}
                      <Show when={nb.pub_year !== undefined}>
                        <span class="small-text" style={{ opacity: 0.4, "flex-shrink": "0" }}>{nb.pub_year}</span>
                      </Show>
                    </button>
                  );
                }}
              </For>
            </div>
          </Show>

          {/* No event selected: show global neighbour summary as chips */}
          <Show when={!selectedEvent()}>
            <div style={{ padding: "0.75rem", display: "flex", "flex-wrap": "wrap", gap: "0.4rem", "align-content": "flex-start" }}>
              <For each={sortedGlobalNeighbours()}>
                {(summary) => {
                  const maxCount = sortedGlobalNeighbours()[0]?.eventCount ?? 1;
                  const isFocus = () => focusToken() === summary.token;
                  const sizePx = 11 + (summary.eventCount / maxCount) * 10;
                  const sparklineData = toSeries(tokenTemporalProfile(), summary.token);

                  return (
                    <button
                      class={`chip ${ isFocus() ? "primary" : "" }`}
                      style={{ "font-size": `${ sizePx.toFixed(1) }px`, cursor: "pointer" }}
                      onClick={() => setFocusToken((prev) => prev === summary.token ? null : summary.token)}
                    >
                      <span>{summary.token}</span>

                      <Show when={yearMode() == "range"}>
                        <Sparkline data={sparklineData} color={isFocus() ? "var(--on-primary)" : "var(--tertiary)"} />
                      </Show>

                      <span class="small-text medium-opacity">
                        {summary.eventCount}
                      </span>
                    </button>
                  );
                }}
              </For>
            </div>
          </Show>
        </section>

        {/* RIGHT: documents */}
        <aside class="s3 surface-container" >
          <div class="padding small-text bold" style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}>
            <Show when={focusToken()} fallback="Documents">
              <span>
                Documents for "{focusToken()}"
                <span class="small-text left-padding medium-opacity">
                  {rightPanelDocs().length}
                </span>
              </span>
            </Show>
          </div>

          <Show
            when={rightPanelDocs().length > 0}
            fallback={
              <div class="padding small-opacity small-text">
                Select an event or click a neighbour token
              </div>
            }
          >
            <div style={{ padding: "0.5rem" }}>
              <For each={rightPanelDocs()}>
                {({ docId, year }) => (
                  <button
                    class="chip small-margin"
                    style={{ display: "flex", "justify-content": "space-between", width: "calc(100% - 0.5rem)", cursor: "pointer" }}
                    onClick={() => showDocument(docId)}
                  >
                    <span style={{ "font-family": "'IBM Plex Mono', monospace", "font-size": "0.78rem", overflow: "hidden", "text-overflow": "ellipsis" }}>
                      {docId}
                    </span>
                    <Show when={year !== undefined}>
                      <span class="small-text medium-opacity" style={{ "flex-shrink": "0", "padding-left": "0.4rem" }}>
                        {year}
                      </span>
                    </Show>
                  </button>
                )}
              </For>
            </div>
          </Show>
        </aside>

      </div>

      {/* Footer */}
      <footer
        class="fixed max center-align small-padding surface-container-low"
        style={{ "flex-shrink": "0" }}
      >
        {yearFiltered().length} events
        {" • "}
        {neighbourIndex().size} event-linked tokens
        {" • "}
        {rightPanelDocs().length} documents
        <Show when={focusToken()}>
          {" • "}
          focus: "{focusToken()}"
          {" "}
          ({neighbourIndex().get(focusToken()!)?.eventCount ?? 0} events,
          {" "}
          {neighbourIndex().get(focusToken()!)?.docIds.size ?? 0} docs)
        </Show>
        <Show when={fromYear() !== yearBounds()[0] || toYear() !== yearBounds()[1]}>
          {" • "}
          {fromYear()}–{toYear()}
        </Show>
      </footer>

    </div>
  );
};

export default NeighbourhoodBrowser;