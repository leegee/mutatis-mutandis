import {
  children,
  createMemo, For, Show,
  type ParentComponent
} from "solid-js";
import { controls } from "../state/controls.store";
import { controlsActions as A } from "../state/controls.actions";
import { tier2Data } from "../state/tier2data.store";
import type { ViewMode, YearMode } from "../state/controls.store";
import { getYearBounds, getYearFiltered } from "../state/selectors";

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
  const conceptsMemo = createMemo(() => Object.keys(tier2Data));
  const yearBoundsMemo = createMemo(() => getYearBounds());

  return (
    <header class="center-align max surface-container-low small-padding top-padding">
      <nav>
        <div class="field suffix border middle-align">
          <select
            value={controls.concept}
            onChange={(e) => A.setConcept(e.currentTarget.value)}
          >
            <For each={conceptsMemo()}>
              {(c) => <option value={c}>{c}</option>}
            </For>
          </select>
          <output>Concept</output>
        </div>

        <Show when={fdgControls()}>
          <div class="field suffix border middle-align">
            <select
              value={controls.viewMode}
              onChange={(e) =>
                A.setViewMode(e.currentTarget.value as ViewMode)
              }
            >
              <option value="aggregated">Aggregated</option>
              <option value="events">Events</option>
            </select>
            <output>View</output>
          </div>

          {/* Max hubs */}
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
          <output class="small-padding top-padding">
            Top N {controls.topN}
          </output>
        </div>

        {/* Hub spread */}
        <Show when={props.includeHubSpread}>
          <div class="field middle-align">
            <div class="slider tiny">
              <input
                type="range"
                min={0.2}
                max={2.0}
                step={0.05}
                value={controls.hubSpread}
                onInput={(e) =>
                  A.setHubSpread(Number(e.currentTarget.value))
                }
              />
              <span />
              <span class="tooltip bottom" />
            </div>

            <output class="small-padding top-padding">
              Hub spread {controls.hubSpread.toFixed(2)}
            </output>
          </div>
        </Show>

        {/* Min similarity */}
        <Show when={fdgControls() && controls.viewMode === "aggregated"}>
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
          </div>
        </Show>

        {/* Year mode */}
        <div class="field suffix border middle-align">
          <select
            value={controls.yearMode}
            onChange={(e) =>
              A.setYearMode(
                e.currentTarget.value as YearMode,
                yearBoundsMemo()
              )
            }
          >
            <option value="single">Single year</option>
            <option value="range">Year range</option>
          </select>
          <output>Year mode</output>
        </div>

        {/* Single year mode */}
        <Show when={controls.yearMode === "single"}>
          <nav class="no-space">

            <button
              class="circle chip secondary no-space large-margin bottom-margin"
              onClick={() => A.stepYear(-1)}
            >
              <i>remove</i>
            </button>

            <div class="field middle-align">
              <div class="slider tiny">
                <input
                  type="range"
                  min={yearBoundsMemo()[0]}
                  max={yearBoundsMemo()[1]}
                  step={1}
                  value={controls.fromYear}
                  onInput={(e) =>
                    A.setSingleYear(Number(e.currentTarget.value))
                  }
                />
                <span class="tooltip bottom" />
              </div>

              <output class="small-padding top-padding">
                {controls.fromYear} ({getYearFiltered().length} events)
              </output>
            </div>

            <button
              class="circle chip secondary no-space large-margin bottom-margin"
              onClick={() => A.stepYear(+1)}
            >
              <i>add</i>
            </button>

          </nav>
        </Show>

        {/* Range mode */}
        <Show when={controls.yearMode === "range"}>
          <div class="field middle-align">
            <div class="slider tiny">

              <input
                type="range"
                min={yearBoundsMemo()[0]}
                max={yearBoundsMemo()[1]}
                step={1}
                value={controls.fromYear}
                onInput={(e) =>
                  A.setRange(
                    Math.min(
                      Number(e.currentTarget.value),
                      controls.toYear
                    ),
                    controls.toYear
                  )
                }
              />

              <input
                type="range"
                min={yearBoundsMemo()[0]}
                max={yearBoundsMemo()[1]}
                step={1}
                value={controls.toYear}
                onInput={(e) =>
                  A.setRange(
                    controls.fromYear,
                    Math.max(
                      Number(e.currentTarget.value),
                      controls.fromYear
                    )
                  )
                }
              />

              <span />
              <span class="tooltip bottom" />
              <span class="tooltip bottom" />

            </div>

            <output class="small-padding top-padding">
              <span>
                {controls.fromYear}–{controls.toYear}
              </span>
              <Show when={props.totalEvents}>
                <span class="left-padding">
                  {getYearFiltered().length} / {props.totalEvents!()} events
                </span>
              </Show>
            </output>
          </div>
        </Show>

        {resolved()}

      </nav>
    </header>
  );
}

export default ControlsHeader;
