import {
  children,
  createResource,
  For,
  Show,
  type ParentComponent,
} from "solid-js";
import { controls, MAX_TOP_N } from "../state/controls.store";
import { controlsActions as A } from "../state/controls.actions";
import { queryConcepts } from "../services/db";

import "./ControlsHeader.css";
import { YearTimeline } from "./Graph2/YearTimeline";

interface Props {
  children?: any;
  title?: string;
  includeHubSpread?: boolean;
  fdgControls?: boolean;
  totalEvents?: () => number;
  noYears?: boolean;
}

const ControlsHeader: ParentComponent<Props> = (props) => {
  const resolved = children(() => props.children);
  const fdgControls = () => props.fdgControls ?? true;

  // Concepts list - refetches if dbReady changes (i.e. once, on init)
  const [conceptsResource] = createResource(queryConcepts);
  const concepts = () => conceptsResource() ?? [];

  return (
    <header class="left-align max surface-container-low tiny-padding bottom-padding top-padding no-margin">
      <nav>
        <div class="field suffix border middle-align small">
          <Show when={concepts().length > 0}>
            <select
              value={controls.concept}
              onChange={(e) => A.setConcept(e.currentTarget.value)}
            >
              <For each={concepts()}>{(c) => <option value={c}>{c}</option>}</For>
            </select>
          </Show>
          <span class="tooltip bottom">Concept</span>
        </div>

        <hr class="divider vertical max no-margin no-padding" />

        <YearTimeline tooltipPosition="bottom" />

        <hr class="divider vertical max no-margin no-padding" />

        {/* Top N  .field>:is(input,select) */}
        <div class="field middle-align prefix border small">
          <i class="tiny">tenancy</i>
          <input
            type="number"
            min={1}
            max={MAX_TOP_N}
            step={1}
            value={controls.topN}
            onInput={(e) => A.setTopN(Number(e.currentTarget.value))}
          />
          <span class="tooltip bottom">
            Top Neighbours {controls.topN}.
            <br />
            The number of top-ranked neighbours to display.
            <br />
            Reduce if the graph is too cluttered to read.
          </span>
        </div>

        <Show when={fdgControls()}>
          {/* <div class="field suffix border middle-align">
            <hr class="divider vertical max no-margin no-padding" />

            <select
              value={controls.viewMode}
              onChange={(e) => A.setViewMode(e.currentTarget.value as ViewMode)}
            >
              <option value="aggregated">Aggregated</option>
              <option value="events">Events</option>
            </select>
            <output>View</output>
            <span class="tooltip bottom">
              Switches between aggregated hub-based view and raw event-level
              graph view
            </span>
          </div>

          <hr class="divider vertical max no-margin no-padding" /> */}

          <Show when={controls.viewMode === "aggregated"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range"
                  min={0.01}
                  max={0.95}
                  step={0.05}
                  value={controls.minSimilarity}
                  onInput={(e) =>
                    A.setMinSimilarity(Number(e.currentTarget.value))
                  }
                />
                <span />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                Min sim {controls.minSimilarity.toFixed(2)}
              </output>
              <span class="tooltip bottom">
                Filters out weak connections by only showing relationships whose
                <br />
                similarity score is above the selected threshold.
              </span>
            </div>
          </Show>

          <Show when={controls.viewMode === "aggregated"}>
            <div class="field suffix border middle-align">
              <select
                value={controls.maxHubs}
                onChange={(e) => A.setMaxHubs(Number(e.currentTarget.value))}
              >
                <For each={[10, 20, 50, 100]}>
                  {(n) => <option value={n}>{n}</option>}
                </For>
              </select>
              <output>Max hubs</output>
              <span class="tooltip bottom">
                The maximum number of hubs to generate
              </span>
            </div>
          </Show>
        </Show>

        <hr class="divider vertical max no-margin no-padding" />

        <Show when={props.includeHubSpread}>
          <div class="field middle-align">
            <div class="slider tiny">
              <input
                type="range"
                min={0.2}
                max={2.0}
                step={0.05}
                value={controls.hubSpread}
                onInput={(e) => A.setHubSpread(Number(e.currentTarget.value))}
              />
              <span />
              <span class="tooltip bottom" />
            </div>
            <span class="tooltip bottom">
              Hub spread {controls.hubSpread.toFixed(2)}<br />
              Controls how strongly hub nodes repel each other, affecting how
              clustered or spread out the layout becomes.
            </span>
          </div>
        </Show>

        {resolved()}
      </nav>
    </header>
  );
};

export default ControlsHeader;
