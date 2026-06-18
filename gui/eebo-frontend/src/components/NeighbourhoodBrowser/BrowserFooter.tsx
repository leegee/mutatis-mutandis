/**
 * BrowserFooter.tsx
 *
 * Status bar: event / token / document counts plus active filters.
 */

import { Show, type Component } from "solid-js";
import type { SqliteEventWithNeighbours } from "../../types";
import type { NeighbourIndex } from "./neighbourUtils";
import { controls } from "../../state/controls.store";

interface Props {
  yearFiltered: () => SqliteEventWithNeighbours[];
  neighbourIndex: () => NeighbourIndex;
  rightPanelDocCount: () => number;
  focusToken: () => string | null;
  yearBounds: () => [number, number];
}

const BrowserFooter: Component<Props> = (props) => {
  const focusSummary = () =>
    props.focusToken() ? props.neighbourIndex().get(props.focusToken()!) : undefined;

  const showYearRange = () =>
    controls.fromYear !== props.yearBounds()[0] ||
    controls.toYear !== props.yearBounds()[1];

  return (
    <footer
      class="fixed max center-align small-padding surface-container-low"
      style={{ "flex-shrink": "0" }}
    >
      {props.yearFiltered().length} events
      {" • "}
      {props.neighbourIndex().size} event-linked tokens
      {" • "}
      {props.rightPanelDocCount()} documents

      <Show when={props.focusToken()}>
        {" • "}
        focus: "{props.focusToken()}"
        {" "}
        ({focusSummary()?.eventCount ?? 0} events,
        {" "}
        {focusSummary()?.docIds.size ?? 0} docs)
      </Show>

      <Show when={showYearRange()}>
        {" • "}
        {controls.fromYear}–{controls.toYear}
      </Show>
    </footer>
  );
};

export default BrowserFooter;
