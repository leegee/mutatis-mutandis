import { createResource, For } from "solid-js";
import { controls } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import { getConcepts } from "../../state/controls.selectors";

export default function SingleConceptSelect() {
  const [conceptsResource] = createResource(
    () => getConcepts(),
  );

  const concepts = () => conceptsResource() ?? [];

  return (
    <div class="field border small no-margin no-padding">
      <select
        value={controls.concept ?? ""}
        onChange={(e) =>
          controlsActions.setConceptSelection([
            e.currentTarget.value,
          ])
        }
      >
        <For each={concepts()}>
          {(concept) => (
            <option value={concept}>
              {concept}
            </option>
          )}
        </For>
      </select>
    </div>
  );
}