// src/App.tsx

import { createResource, createSignal, ErrorBoundary, Match, Show, Switch } from "solid-js";
import { Transition } from "solid-transition-group";
import ConceptGraph from "./components/ConceptGraph/ConceptGraph";
import ConceptGraphGuide from "./components/ConceptGraph/ConceptGraphGuide";
import { loadConceptNeighbours } from "./components/ConceptGraph/loadConceptNeighbours.service";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);

  return (
    <>
      <main class="responsive max">
        <ErrorBoundary fallback={
          (err) => <article>
            <div class="error padding">{err.message}</div>
          </article>
        }>
          <Show when={events()} fallback={
            <article class="responsive">
              <progress></progress>
              <h1>Concept Graph</h1>
              <h2>Loading events...</h2>
            </article>
          }>
            {(data) => <ConceptGraph data={data()} />}
          </Show>
        </ErrorBoundary>
      </main >

      <Transition name="slide-fade">
        {openHelp() && (
          <article class="helpContainer right" style="width: 32rem">
            <ConceptGraphGuide />
          </article>
        )}
      </Transition>

      <button class="border small" onClick={() => setOpenHelp(v => !v)}
        style={{
          position: 'fixed',
          top: '2rem',
          right: '1rem',
          'z-index': '100',
        }}>
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
