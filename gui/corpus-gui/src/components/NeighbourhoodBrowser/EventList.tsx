/**
 * EventList.tsx
 *
 * Left-panel: scrollable list of events.
 */

import { type Component, For, Show } from "solid-js";
import type { NeighbourhoodEvent } from "~/types/neighbourhood";

interface Props {
  events: () => NeighbourhoodEvent[];
  selectedEventId: () => number | null;
  onSelect: (eventId: number) => void;
}

const EventList: Component<Props> = (props) => (
  <nav
    class="s3 surface-container"
    style={{
      "flex-shrink": "0",
      "overflow-y": "auto",
      display: "flex",
      "flex-direction": "column",
    }}
  >
    <div class="padding small-text bold">
      Events
      <span class="right-align small-text left-padding medium-opacity">{props.events().length}</span>
    </div>

    <For each={props.events()}>
      {(event) => {
        const isSelected = () => props.selectedEventId() === event.eventId;

        return (
          <button
            class={`chip tiny-padding left-padding right-padding no-round no-margin ${ isSelected() ? "primary" : "transparent"
              }`}
            type="button"
            onClick={() => props.onSelect(event.eventId)}
          >
            <span class="tooltip top">eid:{event.eventId}</span>

            <Show when={event.pubYear !== null} fallback={<span class="small-text">–</span>}>
              <span class="small-text">{event.pubYear}</span>
            </Show>

            <span class="code">{event.docId}</span>

            <span class="small-text medium-opacity">{event.neighbours.length} neighbours</span>
          </button>
        );
      }}
    </For>
  </nav>
);

export default EventList;
