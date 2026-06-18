/**
 * NeighbourhoodBrowser.tsx
 *
 */

import {
  createSignal,
  createMemo,
  createResource,
  For,
  Show,
  type Component,
  onMount,
  onCleanup,
} from "solid-js";

import type { SqliteEventWithNeighbours, SqliteNeighbour } from "../types";
import { createTokenWindowResource } from "../services/tokenWindowApi";
import { controls } from "../state/controls.store";
import { getYearBounds, getYearFiltered } from "../state/selectors";
import { dbReady, dbError } from "../state/tier2data.store";
import ControlsHeader from "./ControlsHeader";
import Sparkline from "./NeighbourhoodBrowser/SparkLine";
import { showDocument } from "../services/documentApi";

interface NeighbourSummary {
  token: string;
  token_idx: string;
  occurrenceCount: number;
  eventCount: number;
  meanScore: number;
  docIds: Set<string>;
  docYears: Map<string, number | undefined>;
  eventKeys: Set<string>;
}

type NeighbourIndex = Map<string, NeighbourSummary>;

// Pure helpers (unchanged)
function eventKey(event: SqliteEventWithNeighbours, idx: number): string {
  return event.event_id !== undefined ? String(event.event_id) : `idx:${ idx }`;
}

function buildNeighbourIndex(events: SqliteEventWithNeighbours[]): NeighbourIndex {
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
          docYears: new Map(),
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
          summary.docYears.set(event.doc_id, event.pub_year);
        }
      }
    }
  }

  for (const s of index.values()) {
    if (s.occurrenceCount > 0) s.meanScore /= s.occurrenceCount;
  }

  return index;
}

function scoreToOpacity(
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


export default function NeighbourhoodBrowser() {
  const [selectedEventId, setSelectedEventId] = createSignal<string | null>(null);
  const eventButtonRefs = new Map<string, HTMLButtonElement>();
  const [focusToken, setFocusToken] = createSignal<string | null>(null);
  const [rightPanelEvent, setRightPanelEvent] = createSignal<{
    doc_id: string;
    token_idx: number;
  } | null>(null);


  // The resource key is a tuple of the three values that should trigger a
  // refetch.  createResource re-runs the fetcher whenever the key changes.
  //
  // We do NOT reset selectedEventId / focusToken here - that would cause a
  // cascade.  The selected id will simply fail to match in selectedEvent()
  // and fall back to null gracefully.
  const resourceKey = () =>
    [controls.concept, controls.fromYear, controls.toYear] as const;

  const [yearFilteredResource] = createResource(
    resourceKey,
    async ([concept, from, to]) => {
      if (!concept || !dbReady()) return [];
      return getYearFiltered(concept, from, to);
    },
  );

  // Stable empty array so memos don't need to handle undefined.
  const yearFiltered = (): SqliteEventWithNeighbours[] => yearFilteredResource() ?? [];

  //  Year bounds resource (for footer display)
  const [yearBoundsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBounds(concept),
  );

  const yearBounds = (): [number, number] =>
    yearBoundsResource() ?? [controls.fromYear, controls.toYear];

  //  Derived memos
  const neighbourIndex = createMemo<NeighbourIndex>(() =>
    buildNeighbourIndex(yearFiltered()),
  );

  const sortedGlobalNeighbours = createMemo(() =>
    [...neighbourIndex().values()].sort(
      (a, b) => b.eventCount - a.eventCount || b.meanScore - a.meanScore,
    ),
  );

  const selectedEvent = createMemo<{ event: SqliteEventWithNeighbours; key: string } | null>(() => {
    const id = selectedEventId();
    if (!id) return null;
    const events = yearFiltered();
    for (let idx = 0; idx < events.length; idx++) {
      const k = eventKey(events[idx], idx);
      if (k === id) return { event: events[idx], key: k };
    }
    return null;
  });

  const activeWindowEvent = createMemo(() => rightPanelEvent() ?? selectedEvent()?.event ?? null,);
  const [windowText] = createTokenWindowResource(activeWindowEvent);

  const selectedEventNeighbours = createMemo<SqliteNeighbour[]>(() => {
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

  const rightPanelDocs = createMemo<
    Array<{ docId: string; year?: number; token_idx: number }>
  >(() => {
    const focusedToken = focusToken();
    if (focusedToken) {
      const summary = neighbourIndex().get(focusedToken);
      if (!summary) return [];
      return [...summary.docYears.entries()]
        .map(([docId, year]) => ({
          docId,
          year,
          token_idx: Number(summary.token_idx),
        }))
        .sort((a, b) => (a.year ?? 0) - (b.year ?? 0));
    }
    const sel = selectedEvent();
    if (sel?.event.doc_id) {
      return [{
        docId: sel.event.doc_id,
        year: sel.event.pub_year,
        token_idx: sel.event.token_idx,
      }];
    }
    return [];
  });

  const focusEventKeys = createMemo<Set<string>>(() => {
    const ft = focusToken();
    if (!ft) return new Set();
    return neighbourIndex().get(ft)?.eventKeys ?? new Set();
  });

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
        if (!byYear) { byYear = new Map(); map.set(nb.token, byYear); }
        let set = byYear.get(year);
        if (!set) { set = new Set(); byYear.set(year, set); }
        set.add(key);
      }
    }
    return map;
  });

  type TemporalProfile = Map<number, Set<string>>;

  function toSeries(
    profile: Map<string, TemporalProfile>,
    token: string,
  ): { year: number; value: number }[] {
    const byYear = profile.get(token);
    if (!byYear) return [];
    return [...byYear.entries()]
      .map(([year, set]) => ({ year, value: set.size }))
      .sort((a, b) => a.year - b.year);
  }

  //  Keyboard navigation
  const selectedIndex = createMemo(() => {
    const id = selectedEventId();
    if (!id) return -1;
    return yearFiltered().findIndex((e, idx) => eventKey(e, idx) === id);
  });

  function moveSelection(delta: number) {
    const list = yearFiltered();
    const next = selectedIndex() + delta;
    if (next < 0 || next >= list.length) return;
    const key = eventKey(list[next], next);
    setSelectedEventId(key);
    queueMicrotask(() => eventButtonRefs.get(key)?.focus());
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (selectedEventId() == null) return;
    switch (e.key) {
      case "ArrowUp":
      case "ArrowLeft":
        e.preventDefault(); moveSelection(-1); break;
      case "ArrowDown":
      case "ArrowRight":
        e.preventDefault(); moveSelection(1); break;
    }
  };

  onMount(() => window.addEventListener("keydown", handleKeyDown));
  onCleanup(() => window.removeEventListener("keydown", handleKeyDown));

  return (
    <article style={{ display: "flex", "flex-direction": "column", height: "100%", width: "100%" }}>

      <ControlsHeader />

      <Show when={dbError()}>
        <div class="padding error-container" role="alert">
          <span class="small-text">Database error: {dbError()}</span>
        </div>
      </Show>

      <Show when={yearFilteredResource.loading}>
        <div class="padding center-align small-text medium-opacity">
          <progress />
          <span style={{ "margin-left": "0.5rem" }}>Loading events</span>
        </div>
      </Show>

      <div class="grid background no-margin"
        style={{ display: "flex", flex: "1", overflow: "hidden" }}
      >

        {/* LEFT: event list */}
        <nav class="s3 surface-container"
          style={{ "flex-shrink": "0", "overflow-y": "auto", display: "flex", "flex-direction": "column" }}
        >
          <div class="padding small-text bold">
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
                  ref={(el) => eventButtonRefs.set(key(), el)}
                  onClick={() => {
                    setRightPanelEvent(null);
                    setFocusToken(null);
                    setSelectedEventId((prev) => prev === key() ? null : key());
                  }}
                >
                  <span class="tooltip top">eid:{event.event_id}</span>
                  <Show when={event.pub_year !== undefined} fallback={"–"}>
                    <span class="small-text">{event.pub_year}</span>
                  </Show>
                  <span class="code">{event.doc_id ?? key()}</span>
                  <span class="small-text medium-opacity">
                    {event.neighbours.length} neighbours
                  </span>
                </button>
              );
            }}
          </For>
        </nav>

        {/* CENTRE: neighbour tokens */}
        <section class="s6 surface-container"
          style={{ flex: "1", "overflow-y": "auto", display: "flex", "flex-direction": "column" }}
        >
          <Show when={activeWindowEvent()}>
            {(event) => (
              <aside class="center-align small-padding border small-round">
                <Show
                  when={windowText()}
                  fallback={<div><p>Loading context</p><progress /></div>}
                >
                  {(text) => (
                    <>
                      <blockquote innerHTML={text()} />
                      <button
                        class="chip"
                        disabled={!event().doc_id}
                        onClick={() => {
                          if (event().doc_id) showDocument(event().doc_id, event().token_idx)
                        }}
                      >
                        {event().doc_id ?? "No document"}
                      </button>
                    </>
                  )}
                </Show>
              </aside>
            )}
          </Show>

          <div class="padding small-text bold"
            style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}
          >
            <Show when={selectedEvent()}
              fallback={
                <span>
                  Neighbours: all events
                  <span class="small-text left-padding medium-opacity">
                    {sortedGlobalNeighbours().length} tokens
                  </span>
                </span>
              }
            >
              {(sel) => (
                <span>
                  Neighbours: {sel().event.doc_id ?? sel().key}
                  <Show when={sel().event.pub_year !== undefined}>
                    <span class="small-text left-padding medium-opacity">
                      {sel().event.pub_year}
                    </span>
                  </Show>
                  <span class="small-text left-padding medium-opacity">
                    {selectedEventNeighbours().length} tokens
                  </span>
                </span>
              )}
            </Show>
          </div>

          {/* Selected event: neighbour rows with score bar */}
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
                    <button
                      class="responsive max no-round left-padding right-padding"
                      style={{
                        color: "oldlace",
                        display: "flex",
                        "align-items": "center",
                        gap: "0.5rem",
                        background: isFocus() ? "rgba(100,180,255,0.12)" : "transparent",
                        opacity: op,
                        transition: "background 0.1s",
                      }}
                      onClick={() => {
                        setRightPanelEvent(null);
                        setFocusToken((prev) => prev === nb.token ? null : nb.token);
                      }}
                    >
                      <div style={{
                        width: "20%",
                        "flex-shrink": "0",
                        position: "relative",
                        height: "6px",
                        background: "rgba(255,255,255,0.08)",
                        "border-radius": "3px",
                      }}>
                        <div style={{
                          position: "absolute",
                          left: 0,
                          top: 0,
                          height: "100%",
                          width: `${ barPct }%`,
                          background: "rgba(100,180,255,0.7)",
                          "border-radius": "3px",
                        }} />
                      </div>
                      <span style={{
                        "font-family": "'IBM Plex Mono', monospace",
                        "font-size": "0.85rem",
                        flex: "1",
                        "text-align": "left",
                      }}>
                        {nb.token}
                      </span>
                      <span class="small-text" style={{ opacity: 0.55, "flex-shrink": "0" }}>
                        {nb.score.toFixed(4)}
                      </span>
                      <Show when={nb.pub_year !== undefined}>
                        <span class="small-text" style={{ opacity: 0.4, "flex-shrink": "0" }}>
                          {nb.pub_year}
                        </span>
                      </Show>
                    </button>
                  );
                }}
              </For>
            </div>
          </Show>

          {/* No event selected: global neighbour chips */}
          <Show when={!selectedEvent()}>
            <div style={{
              padding: "0.75rem",
              display: "flex",
              "flex-wrap": "wrap",
              gap: "0.4rem",
              "align-content": "flex-start",
            }}>
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
                      onClick={() => {
                        setFocusToken((prev) => {
                          setRightPanelEvent(null);
                          return prev === summary.token ? null : summary.token;
                        });
                      }}
                    >
                      <span>{summary.token}</span>
                      <Show when={controls.yearMode === "range"}>
                        <Sparkline
                          data={sparklineData}
                          color={isFocus() ? "var(--on-primary)" : "var(--tertiary)"}
                        />
                      </Show>
                      <span class="small-text medium-opacity">{summary.eventCount}</span>
                    </button>
                  );
                }}
              </For>
            </div>
          </Show>
        </section>

        {/* RIGHT: documents */}
        <aside class="s3 surface-container">
          <div class="padding small-text bold"
            style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}
          >
            <Show when={focusToken()} fallback="Documents">
              <span>
                Documents for <q>{focusToken()}</q>
                <span class="small-text left-padding medium-opacity">
                  {rightPanelDocs().length}
                </span>
              </span>
            </Show>
          </div>

          <Show when={rightPanelDocs().length > 0}
            fallback={
              <div class="padding small-opacity small-text">
                Select an event or click a neighbour token
              </div>
            }
          >
            <div style={{ padding: "0.5rem" }}>
              <For each={rightPanelDocs()}>
                {({ docId, year, token_idx }) => (
                  <button
                    class="chip small-margin"
                    style={{
                      display: "flex",
                      "justify-content": "space-between",
                      width: "calc(100% - 0.5rem)",
                      cursor: "pointer",
                    }}
                    onClick={() =>
                      setRightPanelEvent((prev) =>
                        prev?.doc_id === docId ? null : { doc_id: docId, token_idx },
                      )
                    }
                  >
                    <span style={{
                      "font-family": "'IBM Plex Mono', monospace",
                      "font-size": "0.78rem",
                      overflow: "hidden",
                      "text-overflow": "ellipsis",
                    }}>
                      {docId}
                    </span>
                    <Show when={year !== undefined}>
                      <span
                        class="small-text medium-opacity"
                        style={{ "flex-shrink": "0", "padding-left": "0.4rem" }}
                      >
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
      <footer class="fixed max center-align small-padding surface-container-low"
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
        <Show when={
          controls.fromYear !== yearBounds()[0] ||
          controls.toYear !== yearBounds()[1]
        }>
          {" • "}
          {controls.fromYear}–{controls.toYear}
        </Show>
      </footer>

    </article>
  );
};

