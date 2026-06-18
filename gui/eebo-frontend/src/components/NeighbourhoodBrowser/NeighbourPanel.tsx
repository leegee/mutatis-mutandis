/**
 * NeighbourPanel.tsx
 *
 * Centre panel: shows either
 *  - the selected event's neighbours as scored rows, or
 *  - all neighbours as sized/sparklined chips when nothing is selected.
 */

import { For, Show, type Component } from "solid-js";
import type { SqliteEventWithNeighbours, SqliteNeighbour } from "../../types";
import type { NeighbourSummary, TemporalProfile, TemporalPoint } from "./neighbourUtils";
import { scoreToOpacity } from "./neighbourUtils";
import { controls } from "../../state/controls.store";
import Sparkline from "./SparkLine";
import ContextAside from "./ContextAside";

// Types

interface SelectedEventInfo {
  event: SqliteEventWithNeighbours;
  key: string;
}

interface Props {
  selectedEvent: () => SelectedEventInfo | null;
  selectedEventNeighbours: () => SqliteNeighbour[];
  selectedScoreRange: () => [number, number];
  sortedGlobalNeighbours: () => NeighbourSummary[];
  focusToken: () => string | null;
  tokenTemporalProfile: () => TemporalProfile;
  toSeries: (profile: TemporalProfile, token: string) => TemporalPoint[];
  windowText: () => string | null | undefined;
  onFocusToken: (token: string) => void;
  rightPanelEvent: () => { doc_id: string; token_idx: number } | null;
}

function PanelHeading(props: {
  selectedEvent: () => SelectedEventInfo | null;
  neighbourCount: () => number;
}) {
  return (
    <div class="padding small-text bold"
      style={{ "border-bottom": "1px solid rgba(255,255,255,0.08)" }}
    >
      <Show when={props.selectedEvent()}
        fallback={
          <span>
            Neighbours: all events
            <span class="small-text left-padding medium-opacity">
              {props.neighbourCount()} tokens
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
              {sel().event.neighbours.length} tokens
            </span>
          </span>
        )}
      </Show>
    </div>
  );
}

// Selected-event rows
function SelectedNeighbourRows(props: {
  neighbours: () => SqliteNeighbour[];
  scoreRange: () => [number, number];
  focusToken: () => string | null;
  onFocusToken: (token: string) => void;
}) {
  return (
    <div style={{ padding: "0.5rem 0" }}>
      <For each={props.neighbours()}>
        {(nb) => {
          const [sMin, sMax] = props.scoreRange();
          const op = scoreToOpacity(nb.score, sMin, sMax);
          const barPct = sMax > sMin
            ? ((nb.score - sMin) / (sMax - sMin)) * 100
            : 100;
          const isFocus = () => props.focusToken() === nb.token;

          return (
            <button class="responsive max no-round left-padding right-padding"
              style={{
                color: "oldlace",
                display: "flex",
                "align-items": "center",
                gap: "0.5rem",
                background: isFocus() ? "rgba(100,180,255,0.12)" : "transparent",
                opacity: op,
                transition: "background 0.1s",
              }}
              onClick={() => props.onFocusToken(nb.token)}
            >
              {/* Score bar */}
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
  );
}

// Global chip cloud

function GlobalNeighbourChips(props: {
  sortedNeighbours: () => NeighbourSummary[];
  focusToken: () => string | null;
  tokenTemporalProfile: () => TemporalProfile;
  toSeries: (profile: TemporalProfile, token: string) => TemporalPoint[];
  onFocusToken: (token: string) => void;
}) {
  return (
    <div style={{
      padding: "0.75rem",
      display: "flex",
      "flex-wrap": "wrap",
      gap: "0.4rem",
      "align-content": "flex-start",
      "justify-content": "space-between",
    }}>
      <For each={props.sortedNeighbours()}>
        {(summary) => {
          const maxCount = props.sortedNeighbours()[0]?.eventCount ?? 1;
          const isFocus = () => props.focusToken() === summary.token;
          const sizePx = 11 + (summary.eventCount / maxCount) * 10;
          const sparklineData = props.toSeries(props.tokenTemporalProfile(), summary.token);

          return (
            <button
              class={`chip ${ isFocus() ? "primary" : "" }`}
              style={{ "font-size": `${ sizePx.toFixed(1) }px`, cursor: "pointer" }}
              onClick={() => props.onFocusToken(summary.token)}
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
  );
}

// Composed panel

const NeighbourPanel: Component<Props> = (props) => {
  const activeEvent = () => props.rightPanelEvent() ?? props.selectedEvent()?.event ?? null;

  return (
    <section
      class="s6 surface-container"
      style={{ flex: "1", "overflow-y": "auto", display: "flex", "flex-direction": "column" }}
    >
      <Show when={activeEvent()}>
        {(event) => (
          <ContextAside event={event} windowText={props.windowText} />
        )}
      </Show>

      <PanelHeading
        selectedEvent={props.selectedEvent}
        neighbourCount={() => props.sortedGlobalNeighbours().length}
      />

      <Show when={props.selectedEvent()}>
        <SelectedNeighbourRows
          neighbours={props.selectedEventNeighbours}
          scoreRange={props.selectedScoreRange}
          focusToken={props.focusToken}
          onFocusToken={props.onFocusToken}
        />
      </Show>

      <Show when={!props.selectedEvent()}>
        <GlobalNeighbourChips
          sortedNeighbours={props.sortedGlobalNeighbours}
          focusToken={props.focusToken}
          tokenTemporalProfile={props.tokenTemporalProfile}
          toSeries={props.toSeries}
          onFocusToken={props.onFocusToken}
        />
      </Show>
    </section>
  );
};

export default NeighbourPanel;
