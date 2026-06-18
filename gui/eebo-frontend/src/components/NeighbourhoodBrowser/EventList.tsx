/**
 * EventList.tsx
 *
 * Left-panel: scrollable list of events. Keyboard-navigable.
 */

import { For, Show, type Component } from "solid-js";
import type { SqliteEventWithNeighbours } from "../../types";
import { eventKey } from "./neighbourUtils";

interface Props {
  events: () => SqliteEventWithNeighbours[];
  selectedEventId: () => string | null;
  focusEventKeys: () => Set<string>;
  focusToken: () => string | null;
  buttonRefs: Map<string, HTMLButtonElement>;
  onSelect: (key: string) => void;
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
      <span class="right-align small-text left-padding medium-opacity">
        {props.events().length}
      </span>
    </div>

    <For each={props.events()}>
      {(event, idx) => {
        const key = (): string => eventKey(event, idx());
        const isSelected = () => props.selectedEventId() === key();
        const hasFocus = () => {
          const ft = props.focusToken();
          return !ft || props.focusEventKeys().has(key());
        };

        return (
          <button
            class={`chip tiny-padding left-padding right-padding no-round no-margin ${ isSelected() ? "primary" : "transparent"
              }`}
            style={{ opacity: hasFocus() ? 1 : 0.35, transition: "opacity 0.15s" }}
            ref={(el) => props.buttonRefs.set(String(key()), el)}
            onClick={() => props.onSelect(key())}
          >
            <span class="tooltip top">eid:{String(event.event_id)}</span>
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
);

export default EventList;