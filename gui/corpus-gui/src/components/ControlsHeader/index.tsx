/** biome-ignore-all lint/a11y/useKeyWithClickEvents: <todo> */
import { children, type ParentComponent, Show } from "solid-js";

import "./ControlsHeader.css";
import SingleConceptSelect from "./SingleConceptSelect";
import { YearTimeline } from "./YearTimeline";

interface Props {
  // biome-ignore lint/suspicious/noExplicitAny: <is ok>
  children?: any;
  title?: string;
  includeHubSpread?: boolean;
  multiConcept?: boolean;
  noYearTimeline?: boolean;
  topN?: boolean;
  totalEvents?: () => number;
  authorMatch?: boolean;
}

const ControlsHeader: ParentComponent<Props> = (props) => {
  const resolved = children(() => props.children);

  return (
    <nav class="toolbar no-round no-margin no-padding">
      <div class="field suffix border middle-align">
        <SingleConceptSelect />
      </div>

      <Show when={!props.noYearTimeline}>
        <hr class="divider vertical max no-margin no-padding" />
        <YearTimeline />
      </Show>

      <hr class="divider vertical max no-margin no-padding" />

      {/* <Show when={props.authorMatch}>
        <div class="field label border small no-margin no-padding author-match">
          <input
            type="search"
            value={controls.authorMatch || ""}
            onChange={(e) => A.setAuthorMatch(e.currentTarget.value)}
          />
          <label>Match Author</label>
          <span class="tooltip bottom">Match authors containing characters entered here</span>
        </div>
      </Show> */}

      {resolved()}
    </nav>
  );
};

export default ControlsHeader;
