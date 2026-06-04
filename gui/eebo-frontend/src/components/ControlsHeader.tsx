import {
  children,
  createResource,
  For,
  Show,
  type ParentComponent,
} from "solid-js";
import { controls, MAX_TOP_N } from "../state/controls.store";
import { controlsActions as A } from "../state/controls.actions";
import type { ViewMode, YearMode } from "../state/controls.store";
import { getYearBounds, getYearFiltered } from "../state/selectors";
import { queryConcepts } from "../services/db";

import "./ControlsHeader.css";
import { Icon } from "./Icon";

interface Props {
  children?: any;
  title?: string;
  includeHubSpread?: boolean;
  fdgControls?: boolean;
  totalEvents?: () => number;
}

const ControlsHeader: ParentComponent<Props> = (props) => {
  const resolved = children(() => props.children);
  const fdgControls = () => props.fdgControls ?? true;

  // Concepts list - refetches if dbReady changes (i.e. once, on init)
  const [conceptsResource] = createResource(queryConcepts);
  const concepts = () => conceptsResource() ?? [];

  // Year bounds - refetches when concept changes
  const [yearBoundsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBounds(concept),
  );
  const yearBounds = (): [number, number] =>
    yearBoundsResource() ?? [controls.fromYear, controls.toYear];

  // Filtered event count - refetches when concept or year range changes
  const [yearFilteredResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredCount = () => yearFilteredResource()?.length ?? 0;

  return (
    <header class="left-align max surface-container-low tiny-padding bottom-padding top-padding no-margin">
      <nav>
        <Icon class="circle large fill no-padding no-margin no-space" />

        {/* <hr style="width: 3em; background: transparent" /> */}

        <div class="field suffix border middle-align small">
          <select
            value={controls.concept}
            onChange={(e) => A.setConcept(e.currentTarget.value)}
          >
            <For each={concepts()}>{(c) => <option value={c}>{c}</option>}</For>
          </select>
          <span class="tooltip bottom">Concept</span>
        </div>

        <hr class="divider vertical max no-margin no-padding" />

        {/* Year mode */}
        <div class="field suffix border middle-align small">
          <select
            value={controls.yearMode}
            onChange={(e) =>
              A.setYearMode(e.currentTarget.value as YearMode, yearBounds())
            }
          >
            <option value="single">Single year</option>
            <option value="range">Year range</option>
          </select>
          <span class="tooltip bottom">
            Show date for one year or a span of years.
          </span>
        </div>

        {/* Single year mode */}
        <Show when={controls.yearMode === "single"}>
          <hr class="divider vertical max no-margin no-padding" />
          <nav class="no-space">
            <button
              class="circle chip tiny no-border"
              onClick={() => A.stepYear(-1)}
            >
              <i>chevron_left</i>
              <span class="tooltip bottom">Retreat by one year</span>
            </button>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range"
                  min={yearBounds()[0]}
                  max={yearBounds()[1]}
                  step={1}
                  value={controls.fromYear}
                  onInput={(e) =>
                    A.setSingleYear(Number(e.currentTarget.value))
                  }
                />
                <span class="tooltip bottom" />
              </div>
              <div class="tooltip bottom">
                {controls.fromYear} ({filteredCount()} events)
              </div>
            </div>
            <button
              class="circle chip tiny no-border"
              onClick={() => A.stepYear(+1)}
            >
              <i>chevron_right</i>
              <span class="tooltip bottom">Advance by one year</span>
            </button>
          </nav>
        </Show>

        {/* Range mode */}
        <Show when={controls.yearMode === "range"}>
          <hr class="divider vertical max no-margin no-padding" />
          <div class="field middle-align">
            <div class="slider tiny">
              <input
                type="range"
                min={yearBounds()[0]}
                max={yearBounds()[1]}
                step={1}
                value={controls.fromYear}
                onInput={(e) =>
                  A.setRange(
                    Math.min(Number(e.currentTarget.value), controls.toYear),
                    controls.toYear,
                  )
                }
              />
              <input
                type="range"
                min={yearBounds()[0]}
                max={yearBounds()[1]}
                step={1}
                value={controls.toYear}
                onInput={(e) =>
                  A.setRange(
                    controls.fromYear,
                    Math.max(Number(e.currentTarget.value), controls.fromYear),
                  )
                }
              />
              <span />
              <span class="tooltip bottom" />
              <span class="tooltip bottom" />
            </div>
            <div class="tooltip bottom">
              {controls.fromYear}&mdash;{controls.toYear}
              <Show when={props.totalEvents}>
                <span class="left-padding">
                  {filteredCount()} / {props.totalEvents!()} events
                </span>
              </Show>
            </div>
          </div>
        </Show>

        <hr class="divider vertical max no-margin no-padding" />

        {/* Top N  .field>:is(input,select) */}
        <div class="field middle-align prefix  border small">
          <i class="tiny">tenancy</i>
          <input
            type="number"
            min={1}
            max={MAX_TOP_N}
            step={1}
            value={controls.topN}
            onInput={(e) => A.setTopN(Number(e.currentTarget.value))}
          />
          {/* <span class="tooltip bottom">
            Top Neighbours {controls.topN}.
            <br />
            The number of top-ranked neighbours to display.
            <br />
            Reduce if the graph is too cluttered to read.
          </span> */}
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
            <output class="small-padding top-padding">
              Hub spread {controls.hubSpread.toFixed(2)}
            </output>
            <span class="tooltip bottom">
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
