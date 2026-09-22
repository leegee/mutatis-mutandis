/**
 * NeighbourhoodBrowser.tsx
 *
 * Root component. Owns event selection and wires the panels.
 * Database access is not part of the component.
 */

import { type Component, Show } from "solid-js";
import BrowserFooter from "./BrowserFooter";
import DocumentPanel from "./DocumentPanel";
import EventList from "./EventList";
import NeighbourPanel from "./NeighbourPanel";
import { useNeighbourhoodState } from "./useNeighbourhoodState";

const NeighbourhoodBrowser: Component = () => {
  console.log("before state");

  const state = useNeighbourhoodState();

  const selectedEvent = () =>
    state.data().events.find(
      (event) => event.eventId === state.selectedEventId(),
    ) ?? null;

  function handleSelectEvent(eventId: number) {
    state.setSelectedEventId((prev) =>
      prev === eventId ? null : eventId,
    );
  }

  return (
    <article
      style={{
        display: "flex",
        "flex-direction": "column",
        height: "100%",
        width: "100%",
      }}
    >
      <Show when={state.error()}>
        <div class="padding error-container" role="alert">
          <span class="small-text">
            Database error: {state.error()}
          </span>
        </div>
      </Show>

      <Show when={state.isLoading()}>
        <div class="padding center-align small-text medium-opacity">
          <progress />
          <span style={{ "margin-left": "0.5rem" }}>
            Loading events
          </span>
        </div>
      </Show>

      <div
        class="grid background no-margin"
        style={{
          display: "flex",
          flex: "1",
          overflow: "hidden",
        }}
      >
        <EventList
          events={() => state.data().events}
          selectedEventId={state.selectedEventId}
          onSelect={handleSelectEvent}
        />

        <NeighbourPanel selectedEvent={selectedEvent} />

        <DocumentPanel selectedEvent={selectedEvent} />
      </div>

      <BrowserFooter data={state.data} />
    </article>
  );
};

export default NeighbourhoodBrowser;
