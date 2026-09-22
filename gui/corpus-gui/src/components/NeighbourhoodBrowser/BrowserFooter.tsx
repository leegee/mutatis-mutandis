/**
 * BrowserFooter.tsx
 *
 * Status bar: event count and active filters.
 */

import { type Component, Show } from "solid-js";
import { controls } from "~/state/controls.store";
import type { NeighbourhoodData } from "~/types/neighbourhood";

interface Props {
  data: () => NeighbourhoodData;
}

const BrowserFooter: Component<Props> = (props) => {
  const showYearRange = () =>
    controls.fromYear !== props.data().yearBounds[0] || controls.toYear !== props.data().yearBounds[1];

  return (
    <footer class="fixed max center-align small-padding surface-container-low" style={{ "flex-shrink": "0" }}>
      {props.data().events.length} events
      <Show when={showYearRange()}>
        {" • "}
        {controls.fromYear}–{controls.toYear}
      </Show>
    </footer>
  );
};

export default BrowserFooter;
