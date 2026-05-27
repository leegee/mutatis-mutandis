// src/App.tsx

import { createResource, createSignal, ErrorBoundary, Match, Show, Switch } from "solid-js";
import { Transition } from "solid-transition-group";
import ConceptGraphGuide from "./components/ConceptGraph/ConceptGraphGuide";
import { loadConceptNeighbours } from "./components/ConceptGraph/loadConceptNeighbours.service";
// import ConceptGraph from "./components/ConceptGraph/ConceptGraph";
// import ConceptGraph from "./components/ConceptGraph2";
// import ConceptGraph from "./components/ConceptGraph3";
// import NeighbourhoodBrowser from "./components/NeighbourhoodBrowser";
// import ContextGraph4, { type Tier2Data } from "./components/ContextGraph4";
import NeighbourhoodBrowser from "./components/NeighbourhoodBrowser";
import ContextGraph5 from "./components/ContextGraph5";

import "./App.css"
import DiachronicChart from "./components/DiachronicChart";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [open, setOpen] = createSignal(false);
  const [view, setView] = createSignal<'graph' | 'table' | 'help' | 'diachronic'>('graph')

  return (
    <>
      <nav id='app-nav' class={`surface-container fill left scroll ${ open() ? 'max' : 'small' }`}>
        <header class="center-align ">
          <button class="extra transparent" onClick={() => setOpen(!open())}>
            <Switch>
              <Match when={open()}>
                <i>menu_open</i>
              </Match>
              <Match when={!open()}>
                <i>menu</i>
              </Match>
            </Switch>
          </button>
        </header>

        <a onClick={() => setView('graph')}>
          <i>graph_5</i>
          <span>Event Graph</span>
        </a>
        <a onClick={() => setView('table')}>
          <i>view_column</i>
          <span>Neighbourhood Table</span>
        </a>
        <a onClick={() => setView('diachronic')}>
          {/* <i>experiment</i> */}
          <i>hourglass</i>
          <span>Diachronic Chart</span>
        </a>

        <hr />

        <a class="" onClick={() => setOpenHelp(!openHelp())}>
          <i>help</i>
          <span>Guide</span>
        </a>
      </nav>

      <main class="responsive max no-padding">
        <ErrorBoundary fallback={
          (err) => <article>
            <div class="error padding">{err.message}</div>
          </article>
        }>
          <Show when={events()} fallback={
            <article class="small-round padding border medium no-padding">
              <div class="padding absolute center middle">
                <h5>Loading events...</h5>
              </div>
              <progress />
            </article>
          }>
            {(data) => (
              <Switch>
                <Match when={view() === 'graph'}>
                  {/* <ContextGraph4 data={data()} /> */}
                  <ContextGraph5 data={data()} />
                </Match>
                <Match when={view() === 'table'}>
                  <NeighbourhoodBrowser data={data()} />
                </Match>
                <Match when={view() === 'diachronic'}>
                  <DiachronicChart data={data()} />
                </Match>
              </Switch>
            )}
          </Show>
        </ErrorBoundary>
      </main >

      <Transition name="slide-fade">
        <Show when={openHelp()}>
          <article class="helpContainer right" style="width: 32rem">
            <Switch fallback={<article>To do...</article>}>
              <Match when={view() === 'graph'}>
                <ConceptGraphGuide />
              </Match>
            </Switch>
          </article>
        </Show>
      </Transition>
    </>
  );
}
