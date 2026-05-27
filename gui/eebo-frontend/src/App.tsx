// src/App.tsx

import {
  createResource,
  createSignal,
  ErrorBoundary,
  lazy,
  Match,
  Show,
  Switch,
} from "solid-js";
import { Transition } from "solid-transition-group";
import ConceptGraphGuide from "./components/ConceptGraph/ConceptGraphGuide";
import { loadConceptNeighbours } from "./components/ConceptGraph/loadConceptNeighbours.service";

const ContextGraph5 = lazy(() => import("./components/ContextGraph5"));
const NeighbourhoodBrowser = lazy(
  () => import("./components/NeighbourhoodBrowser")
);
const DiachronicChart = lazy(
  () => import("./components/DiachronicChart")
);
const CosmosContextGraph = lazy(
  () => import("./components/CosmosContextGraph")
);

import "./App.css";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [open, setOpen] = createSignal(false);

  const [view, setView] = createSignal<
    "graph" | "table" | "help" | "diachronic" | "cosmos"
  >("graph");

  const navItems = [
    { key: "graph", icon: "graph_5", label: "Event FDG" },
    { key: "table", icon: "view_column", label: "Neighbourhood Table" },
    { key: "diachronic", icon: "hourglass", label: "Diachronic Chart" },
    { key: "cosmos", icon: "experiment", label: "Cosmos FDG" },
  ] as const;

  return (
    <>
      <nav
        id="app-nav"
        class={`surface-container fill left scroll ${ open() ? "max" : "small"
          }`}
      >
        <header class="center-align ">
          <button
            class="extra transparent"
            onClick={() => setOpen(!open())}
          >
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

        {navItems.map((item) => (
          <a
            onClick={() => setView(item.key)}
            classList={{
              active: view() === item.key,
            }}
          >
            <i>{item.icon}</i>
            <span>{item.label}</span>
          </a>
        ))}

        <hr />

        <a onClick={() => setOpenHelp(!openHelp())}>
          <i>help</i>
          <span>Guide</span>
        </a>
      </nav>

      <main class="responsive max no-padding">
        <ErrorBoundary
          fallback={(err) => (
            <article>
              <div class="error padding">{err.message}</div>
            </article>
          )}
        >
          <Show
            when={events()}
            fallback={
              <article class="small-round padding border medium no-padding">
                <div class="padding absolute center middle">
                  <h5>Loading events...</h5>
                </div>
                <progress />
              </article>
            }
          >
            {(data) => (
              <Switch>
                <Match when={view() === "graph"}>
                  <ContextGraph5 data={data()} />
                </Match>

                <Match when={view() === "table"}>
                  <NeighbourhoodBrowser data={data()} />
                </Match>

                <Match when={view() === "diachronic"}>
                  <DiachronicChart data={data()} />
                </Match>

                <Match when={view() === "cosmos"}>
                  <CosmosContextGraph data={data()} />
                </Match>
              </Switch>
            )}
          </Show>
        </ErrorBoundary>
      </main>

      <Transition name="slide-fade">
        <Show when={openHelp()}>
          <article class="helpContainer right" style="width: 32rem">
            <Switch fallback={<article>To do...</article>}>
              <Match when={view() === "graph"}>
                <ConceptGraphGuide />
              </Match>
            </Switch>
          </article>
        </Show>
      </Transition>
    </>
  );
}
