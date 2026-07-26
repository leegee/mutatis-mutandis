import { createResource, For } from "solid-js";
import { controls } from "../../state/controls.store";
import { controlsActions } from "../../state/controls.actions";
import { getConcepts } from "../../state/controls.selectors";

export default function SingleConceptSelect() {
  const [conceptsResource] = createResource(
    () => controls.concept,
    () => getConcepts(),
  );
  const concepts = (): string[] => conceptsResource() ?? [];

  return (
    <div class="field border small no-margin no-padding">
      <select value={controls.concept} onChange={(e) => controlsActions.setConceptSelection([e.currentTarget.value])} >
        <For each={concepts()}>{(c) => <option value={c}>{c}</option>}</For>
      </select>
    </div>
  )
}