// src/App.tsx

import { createResource, createSignal, ErrorBoundary, Match, Show, Switch } from "solid-js";
import { Transition } from "solid-transition-group";
import ConceptGraphGuide from "./components/ConceptGraph/ConceptGraphGuide";
import { loadConceptNeighbours } from "./components/ConceptGraph/loadConceptNeighbours.service";
// import ConceptGraph from "./components/ConceptGraph/ConceptGraph";
// import ConceptGraph from "./components/ConceptGraph2";
// import ConceptGraph from "./components/ConceptGraph3";
// import NeighbourhoodBrowser from "./components/NeighbourhoodBrowser";
import ContextGraph4, { type Tier2Data } from "./components/ContextGraph4";
import NeighbourhoodBrowser from "./components/NeighbourhoodBrowser";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [view, setView] = createSignal<'graph' | 'table' | 'help'>('graph')

  return (
    <>
      <nav class="scroll max left">
        <header>
          <button class="extra circle transparent">
            <i>menu_open</i>
          </button>
        </header>

        <a onClick={() => setView('graph')}>
          <i>graph_5</i>
          <span>Graph</span>
        </a>
        <a onClick={() => setView('table')}>
          <i>table</i>
          <span>Table</span>
        </a>
        <a onClick={() => setOpenHelp(!openHelp())}>
          <i>help</i>
          <span>Guide</span>
        </a>
      </nav>

      <main class="responsive max">
        <ErrorBoundary fallback={
          (err) => <article>
            <div class="error padding">{err.message}</div>
          </article>
        }>
          <Show when={events()} fallback={
            <article class="responsive">
              <progress></progress>
              <h1>Loading events...</h1>
            </article>
          }>
            {(data) => (
              <Switch>
                <Match when={view() === 'graph'}>
                  <ContextGraph4 data={data()} />
                </Match>
                <Match when={view() === 'table'}>
                  <NeighbourhoodBrowser data={data()} />
                </Match>
              </Switch>
            )}
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
    </>
  );
}
