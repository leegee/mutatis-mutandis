/**
 * DocumentPanel.tsx
 *
 * Right panel: lists neighbour occurrences associated with the selected event.
 */

import { type Component, For, Show } from "solid-js";
import type { NeighbourhoodEvent } from "~/types/neighbourhood";

interface Props {
  selectedEvent: () => NeighbourhoodEvent | null;
}

const DocumentPanel: Component<Props> = (props) => (
  <aside class="s3 surface-container" style={{ "z-index": "unset" }}>
    <div class="padding small-text bold">
      Documents
      <Show when={props.selectedEvent()}>
        {(event) => <span class="small-text left-padding medium-opacity">{event().neighbours.length}</span>}
      </Show>
    </div>

    <Show when={props.selectedEvent()} fallback={<div class="padding medium-opacity small-text">Select an event</div>}>
      {(event) => (
        <div class="small-padding">
          <For each={event().neighbours}>
            {(neighbour) => (
              <div
                class="chip small-margin"
                style={{
                  display: "flex",
                  "justify-content": "space-between",
                  width: "calc(100% - 0.5rem)",
                }}
              >
                <span
                  style={{
                    "font-family": "'IBM Plex Mono', monospace",
                    "font-size": "0.78rem",
                    overflow: "hidden",
                    "text-overflow": "ellipsis",
                  }}
                >
                  {neighbour.docId}
                </span>

                <Show when={neighbour.pubYear !== null}>
                  <span
                    class="small-text medium-opacity"
                    style={{
                      "flex-shrink": "0",
                      "padding-left": "0.4rem",
                    }}
                  >
                    {neighbour.pubYear}
                  </span>
                </Show>
              </div>
            )}
          </For>
        </div>
      )}
    </Show>
  </aside>
);

export default DocumentPanel;
