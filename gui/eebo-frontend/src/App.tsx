// src/App.tsx

import { createResource, createSignal, ErrorBoundary, Match, Show, Switch } from "solid-js";
import { Transition } from "solid-transition-group";
import { loadConceptNeighbours } from "./services/loadConceptNeighbours";
import ConceptGraph from "./components/ConceptGraph";
import ConceptGraphGuide from "./components/ConceptGraphGuide";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);

  return (
    <>
      <main class="responsive max">
        <ErrorBoundary fallback={
          (err) => <article><div class="error padding">{err.message}</div></article>
        }>
          <Show when={events()} fallback="Loading events...">
            {(data) => <ConceptGraph data={data()} />}
          </Show>
        </ErrorBoundary>
      </main>

      <Transition name="slide-fade">
        {openHelp() && (
          <article class="helpContainer right" style="width: 32rem">
            <ConceptGraphGuide />
          </article>
        )}
      </Transition>

      <button class="border small" onClick={() => setOpenHelp(v => !v)}>
        <Switch>
          <Match when={!openHelp()}>
            <i>help</i>
          </Match>
          <Match when={openHelp()}>
            <i>close</i>
          </Match>
        </Switch>
      </button>
    </>
  );
}
