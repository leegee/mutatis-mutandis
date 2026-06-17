import {
  children,
  createResource,
  For,
  Show,
  type ParentComponent,
} from "solid-js";
import { controls, MAX_TOP_N } from "../../state/controls.store";
import { controlsActions as A } from "../../state/controls.actions";
import { listConcepts } from "../../services/db";

import "./ControlsHeader.css";
import { YearTimeline } from "./YearTimeline";
import MultiCreatableSelect from "./CreatableSelect";

interface Props {
  children?: any;
  title?: string;
  includeHubSpread?: boolean;
  multiConcept?: boolean;
  noYearTimeline?: boolean;
  topN?: boolean;
  totalEvents?: () => number;
}

const ControlsHeader: ParentComponent<Props> = (props) => {
  const resolved = children(() => props.children);

  // Concepts list - refetches if dbReady changes (i.e. once, on init)
  const [conceptsResource] = createResource(listConcepts);
  const concepts = () => conceptsResource() ?? [];

  return (
    <header class="left-align max surface-container-low tiny-padding bottom-padding top-padding no-margin">
      <nav>
        <div class="field suffix border middle-align small">
          <Show when={concepts().length > 0}>
            <Show when={props.multiConcept} fallback={
              <select value={controls.concept} onChange={(e) => A.setConcept(e.currentTarget.value)} >
                <For each={concepts()}>{(c) => <option value={c}>{c}</option>}</For>
              </select>
            }>

              <div class="row no-space">

                <MultiCreatableSelect
                  selected={controls.conceptSelection}
                  options={concepts()}
                  onChange={A.setConceptSelection}
                  onCreateOption={() => Promise.resolve(false)}
                />

                <div class="no-round bottom">
                  <button class="transparent circle">
                    <i>more_vert</i>
                  </button>
                  <menu class="no-round  bottom left no-wrap">
                    <li onClick={() => A.setConceptSelection(concepts())}>
                      <i>select_all</i>
                      <span>Select all</span>
                    </li>
                    <li onClick={() => A.setConceptSelection([])}>
                      <i>deselect</i>
                      <span>Select none</span>
                    </li>
                    <li onClick={() => A.setConceptSelection(concepts().filter(c => !controls.conceptSelection.includes(c)))}>
                      <i>published_with_changes</i>
                      <span>Invert</span>
                    </li>
                  </menu>
                  <div class="tooltip right">More...</div>
                </div>

              </div>

            </Show>
          </Show>
        </div>

        <Show when={!props.noYearTimeline}>
          <hr class="divider vertical max no-margin no-padding" />
          <YearTimeline tooltipPosition="bottom" />
        </Show>


        <hr class="divider vertical max no-margin no-padding" />

        <Show when={props.topN}>
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
        </Show>

        <Show when={props.includeHubSpread}>

          <hr class="divider vertical max no-margin no-padding" />

          <div class="field middle-align">
            <div class="slider tiny">
              <input type="range" min={0.2} max={2.0} step={0.05}
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
    </header >
  );
};

export default ControlsHeader;
