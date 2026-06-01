import {
  children,
  createResource,
  For, Show,
  type ParentComponent,
} from "solid-js";
import { controls } from "../state/controls.store";
import { controlsActions as A } from "../state/controls.actions";
import type { ViewMode, YearMode } from "../state/controls.store";
import { getYearBounds, getYearFiltered } from "../state/selectors";
import { queryConcepts } from "../services/db";

import "./ControlsHeader.css";

interface Props {
  children?: any;
  title?: string;
  includeHubSpread?: boolean;
  fdgControls?: boolean;
  totalEvents?: () => number;
}

const MAX_TOP_N = 100;

const ControlsHeader: ParentComponent<Props> = (props) => {
  const resolved = children(() => props.children);
  const fdgControls = () => props.fdgControls ?? true;

  // Concepts list — refetches if dbReady changes (i.e. once, on init)
  const [conceptsResource] = createResource(queryConcepts);
  const concepts = () => conceptsResource() ?? [];

  // Year bounds — refetches when concept changes
  const [yearBoundsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBounds(concept),
  );
  const yearBounds = (): [number, number] =>
    yearBoundsResource() ?? [controls.fromYear, controls.toYear];

  // Filtered event count — refetches when concept or year range changes
  const [yearFilteredResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredCount = () => yearFilteredResource()?.length ?? 0;

  return (
    <header class="left-align max surface-container-low small-padding top-padding">
      <nav>
        <hr style="width: 3em; background: transparent" />

        <div class="field suffix border middle-align">
          <select
            value={controls.concept}
            onChange={(e) => A.setConcept(e.currentTarget.value)}
          >
            <For each={concepts()}>
              {(c) => <option value={c}>{c}</option>}
            </For>
          </select>
          <output>Concept</output>
        </div>

        <hr class="divider vertical max no-margin no-padding" />

        {/* Year mode */}
        <div class="field suffix border middle-align">
          <select
            value={controls.yearMode}
            onChange={(e) =>
              A.setYearMode(e.currentTarget.value as YearMode, yearBounds())
            }
          >
            <option value="single">Single year</option>
            <option value="range">Year range</option>
          </select>
          <output>Year mode</output>
        </div>

        {/* Single year mode */}
        <Show when={controls.yearMode === "single"}>
          <hr class="divider vertical max no-margin no-padding" />
          <nav class="no-space">
            <button
              class="circle chip tiny no-space large-margin bottom-margin"
              onClick={() => A.stepYear(-1)}
            >
              <i>remove</i>
            </button>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range"
                  min={yearBounds()[0]}
                  max={yearBounds()[1]}
                  step={1}
                  value={controls.fromYear}
                  onInput={(e) => A.setSingleYear(Number(e.currentTarget.value))}
                />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                {controls.fromYear} ({filteredCount()} events)
              </output>
            </div>
            <button
              class="circle chip tiny no-space large-margin bottom-margin"
              onClick={() => A.stepYear(+1)}
            >
              <i>add</i>
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
            <output class="small-padding top-padding">
              <span>{controls.fromYear}–{controls.toYear}</span>
              <Show when={props.totalEvents}>
                <span class="left-padding">
                  {filteredCount()} / {props.totalEvents!()} events
                </span>
              </Show>
            </output>
          </div>
        </Show>

        {/* Top N */}
        <div class="field middle-align">
          <div class="slider tiny">
            <input
              type="range"
              min={1}
              max={MAX_TOP_N}
              step={1}
              value={controls.topN}
              onInput={(e) => A.setTopN(Number(e.currentTarget.value))}
            />
            <span />
            <span class="tooltip bottom" />
          </div>
          <output class="small-padding top-padding">Top N {controls.topN}</output>
        </div>

        <Show when={fdgControls()}>
          <hr class="divider vertical max no-margin no-padding" />

          <div class="field suffix border middle-align">
            <select
              value={controls.viewMode}
              onChange={(e) => A.setViewMode(e.currentTarget.value as ViewMode)}
            >
              <option value="aggregated">Aggregated</option>
              <option value="events">Events</option>
            </select>
            <output>View</output>
          </div>

          <hr class="divider vertical max no-margin no-padding" />

          <Show when={controls.viewMode === "aggregated"}>
            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range"
                  min={0.01}
                  max={0.95}
                  step={0.05}
                  value={controls.minSimilarity}
                  onInput={(e) => A.setMinSimilarity(Number(e.currentTarget.value))}
                />
                <span />
                <span class="tooltip bottom" />
              </div>
              <output class="small-padding top-padding">
                Min sim {controls.minSimilarity.toFixed(2)}
              </output>
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
          </div>
        </Show>

        {resolved()}
      </nav>
    </header>
  );
};

export default ControlsHeader;