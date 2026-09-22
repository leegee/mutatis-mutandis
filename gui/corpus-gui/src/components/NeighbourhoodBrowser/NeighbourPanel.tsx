/**
 * NeighbourPanel.tsx
 *
 * Centre panel: shows the neighbours of the selected event.
 */

import { type Component, For, Show } from "solid-js";
import type { Neighbour, NeighbourhoodEvent } from "~/types/neighbourhood";

interface Props {
  selectedEvent: () => NeighbourhoodEvent | null;
}

function scoreOpacity(score: number, min: number, max: number): number {
  if (max <= min) {
    return 1;
  }

  return 0.35 + ((score - min) / (max - min)) * 0.65;
}

function SelectedNeighbourRows(props: { neighbours: () => Neighbour[] }) {
  const scoreRange = () => {
    const scores = props.neighbours().map((neighbour) => neighbour.score);

    if (scores.length === 0) {
      return [0, 1] as [number, number];
    }

    return [Math.min(...scores), Math.max(...scores)] as [number, number];
  };

  return (
    <div style={{ padding: "0.5rem 0" }}>
      <For each={props.neighbours()}>
        {(neighbour) => {
          const [scoreMin, scoreMax] = scoreRange();
          const barPct = scoreMax > scoreMin ? ((neighbour.score - scoreMin) / (scoreMax - scoreMin)) * 100 : 100;

          return (
            <div
              class="responsive max no-round left-padding right-padding"
              style={{
                display: "flex",
                "align-items": "center",
                gap: "0.5rem",
                opacity: scoreOpacity(neighbour.score, scoreMin, scoreMax),
              }}
            >
              <div
                style={{
                  width: "20%",
                  "flex-shrink": "0",
                  position: "relative",
                  height: "6px",
                  background: "rgba(255,255,255,0.08)",
                  "border-radius": "3px",
                }}
              >
                <div
                  style={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    height: "100%",
                    width: `${ barPct }%`,
                    background: "rgba(100,180,255,0.7)",
                    "border-radius": "3px",
                  }}
                />
              </div>

              <span
                style={{
                  "font-family": "'IBM Plex Mono', monospace",
                  "font-size": "0.85rem",
                  flex: "1",
                }}
              >
                {neighbour.token}
              </span>

              <span
                class="small-text"
                style={{
                  opacity: 0.55,
                  "flex-shrink": "0",
                }}
              >
                {neighbour.score.toFixed(4)}
              </span>

              <Show when={neighbour.pubYear !== null}>
                <span
                  class="small-text"
                  style={{
                    opacity: 0.4,
                    "flex-shrink": "0",
                  }}
                >
                  {neighbour.pubYear}
                </span>
              </Show>
            </div>
          );
        }}
      </For>
    </div>
  );
}

const NeighbourPanel: Component<Props> = (props) => (
  <section
    class="s6 surface-container"
    style={{
      flex: "1",
      "overflow-y": "auto",
      display: "flex",
      "flex-direction": "column",
    }}
  >
    <div
      class="padding small-text bold"
      style={{
        "border-bottom": "1px solid rgba(255,255,255,0.08)",
      }}
    >
      <Show when={props.selectedEvent()} fallback="Select an event">
        {(event) => (
          <>
            <span>Neighbours: {event().docId}</span>
            <span class="small-text left-padding medium-opacity">{event().pubYear}</span>
            <span class="small-text left-padding medium-opacity">{event().neighbours.length} neighbours</span>
          </>
        )}
      </Show>
    </div>

    <Show
      when={props.selectedEvent()}
      fallback={<div class="padding medium-opacity">Select an event to inspect its semantic neighbours.</div>}
    >
      {(event) => <SelectedNeighbourRows neighbours={() => event().neighbours} />}
    </Show>
  </section>
);

export default NeighbourPanel;
