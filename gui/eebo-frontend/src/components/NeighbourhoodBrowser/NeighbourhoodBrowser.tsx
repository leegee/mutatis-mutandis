/**
 * NeighbourhoodBrowser.tsx
 *
 * Root component. Wires the state hook to the panel layout.
 * All reactive logic lives in useNeighbourhoodState.
 * All rendering logic lives in the panel sub-components.
 */

import { Show, type Component } from "solid-js";
import { dbError } from "../../state/tier2data.store";
import ControlsHeader from "../ControlsHeader";
import { useNeighbourhoodState } from "./useNeighbourhoodState";
import EventList from "./EventList";
import NeighbourPanel from "./NeighbourPanel";
import DocumentPanel from "./DocumentPanel";
import BrowserFooter from "./BrowserFooter";

const NeighbourhoodBrowser: Component = () => {
  const state = useNeighbourhoodState();

  // Consolidated callbacks that keep signal-clearing logic in one place
  function handleSelectEvent(key: string) {
    state.setRightPanelEvent(null);
    state.setFocusToken(null);
    state.setSelectedEventId((prev) => (prev === key ? null : key));
  }

  function handleFocusToken(token: string) {
    state.setRightPanelEvent(null);
    state.setFocusToken((prev) => (prev === token ? null : token));
  }

  function handleSelectDoc(docId: string, tokenIdx: number) {
    state.setRightPanelEvent((prev) =>
      prev?.doc_id === docId ? null : { doc_id: docId, token_idx: tokenIdx },
    );
  }

  return (
    <article style={{ display: "flex", "flex-direction": "column", height: "100%", width: "100%" }}>

      <ControlsHeader />

      <Show when={dbError()}>
        <div class="padding error-container" role="alert">
          <span class="small-text">Database error: {dbError()}</span>
        </div>
      </Show>

      <Show when={state.isLoading()}>
        <div class="padding center-align small-text medium-opacity">
          <progress />
          <span style={{ "margin-left": "0.5rem" }}>Loading events</span>
        </div>
      </Show>

      <div
        class="grid background no-margin"
        style={{ display: "flex", flex: "1", overflow: "hidden" }}
      >
        <EventList
          events={state.yearFiltered}
          selectedEventId={state.selectedEventId}
          focusEventKeys={state.focusEventKeys}
          focusToken={state.focusToken}
          buttonRefs={state.eventButtonRefs}
          onSelect={handleSelectEvent}
        />

        <NeighbourPanel
          selectedEvent={state.selectedEvent}
          selectedEventNeighbours={state.selectedEventNeighbours}
          selectedScoreRange={state.selectedScoreRange}
          rightPanelEvent={state.rightPanelEvent}
          sortedGlobalNeighbours={state.sortedGlobalNeighbours}
          focusToken={state.focusToken}
          tokenTemporalProfile={state.tokenTemporalProfile}
          toSeries={state.toSeries}
          windowText={state.windowText}
          onFocusToken={handleFocusToken}
        />

        <DocumentPanel
          docs={state.rightPanelDocs}
          focusToken={state.focusToken}
          rightPanelEvent={state.rightPanelEvent}
          onSelectDoc={handleSelectDoc}
        />
      </div>

      <BrowserFooter
        yearFiltered={state.yearFiltered}
        neighbourIndex={state.neighbourIndex}
        rightPanelDocCount={() => state.rightPanelDocs().length}
        focusToken={state.focusToken}
        yearBounds={state.yearBounds}
      />

    </article>
  );
};

export default NeighbourhoodBrowser;
