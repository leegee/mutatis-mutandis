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

import "./App.css";

import { loadConceptNeighbours } from "./services/loadConceptNeighbours.service";

import ConceptGraphGuide from "./components/SvgConceptGraph/Guide";
const CosmosContextGraphGuide = lazy(() => import("./components/CosmosContextGraph/Guide"));

const NeighbourhoodBrowser = lazy(() => import("./components/NeighbourhoodBrowser"));
const DiachronicChart = lazy(() => import("./components/DiachronicChart"));
const SvgContextGraph5 = lazy(() => import("./components/SvgConceptGraph"));
const Cosmos = lazy(() => import("./components/CosmosContextGraph/"));

export default function App() {
  const [events] = createResource(loadConceptNeighbours);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [open, setOpen] = createSignal(false);

  const [view, setView] = createSignal<
    "graph" | "table" | "help" | "diachronic" | "cosmos"
  >("cosmos");

  const navItems = [
    { key: "cosmos", icon: "orbit", label: "Force graph (Cosmos GL)" },
    { key: "graph", icon: "graph_5", label: "Force graph (SVG)" },
    { key: "table", icon: "view_column", label: "Neighbourhood Table" },
    { key: "diachronic", icon: "avg_time", label: "Diachronic Chart" },
  ] as const;

  return (
    <>
      <nav id="app-nav" class={`surface-container fill left scroll ${ open() ? "max" : "small" }`} >
        <header class="center-align ">
          <button class="extra transparent" onClick={() => setOpen(!open())} >
            <Switch>
              <Match when={open()}> <i>menu_open</i> </Match>
              <Match when={!open()}> <i>menu</i> </Match>
            </Switch>
          </button>
        </header>

        {navItems.map((item) => (
          <a onClick={() => setView(item.key)} classList={{ active: view() === item.key, }} >
            <i>{item.icon}</i>
            <span>{item.label}</span>
          </a>
        ))}

        <hr class="max" />

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
          <Show when={events()} fallback={<article class="small-round padding border medium no-padding">
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
                  <SvgContextGraph5 data={data()} />
                </Match>

                <Match when={view() === "table"}>
                  <NeighbourhoodBrowser data={data()} />
                </Match>

                <Match when={view() === "diachronic"}>
                  <DiachronicChart data={data()} />
                </Match>

                <Match when={view() === "cosmos"}>
                  <Cosmos data={data()} />
                </Match>

              </Switch>
            )}
          </Show>
        </ErrorBoundary>
      </main>

      <Transition name="slide-fade">
        <Show when={openHelp()}>
          <article class="helpContainer right surface-container-highest padding  high-elevate border">
            <Switch fallback={<article>To do...</article>}>
              <Match when={view() === "graph"}>
                <ConceptGraphGuide />
              </Match>
              <Match when={view() == "cosmos"}>
                <CosmosContextGraphGuide />
              </Match>
            </Switch>
          </article>
        </Show>
      </Transition>
    </>
  );
}
