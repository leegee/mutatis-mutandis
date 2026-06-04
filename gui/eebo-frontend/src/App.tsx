// src/App.tsx

import {
  createSignal,
  ErrorBoundary,
  lazy,
  Match,
  Show,
  Switch,
} from "solid-js";
import { Transition } from "solid-transition-group";

import "./App.css";
import { dbReady, loadTier2Data } from "./state/tier2data.store";
import AppError from "./components/AppError";
import { Icon } from "./components/Icon";

const CosmosContextGraphGuide = lazy(() => import("./components/CosmosContextGraph/Guide"),);
import NeighbourhoodBrowser from "./components/NeighbourhoodBrowser";
import DiachronicChart from "./components/DiachronicChart";
// const Cosmos = lazy(() => import("./components/CosmosContextGraph"));
import Graph2 from "./components/Graph2/Graph2";

export default function App() {
  const [dbLoadingError, setDbLoadingError] = createSignal<string | null>(null);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [open, setOpen] = createSignal(false);
  const [view, setView] = createSignal<
    "table" | "help" | "diachronic" | "cosmos" | "graph2"
  >("graph2");

  try {
    loadTier2Data();
  } catch (e) {
    setDbLoadingError((e as Error).message);
  }

  const navItems = [
    { key: "graph2", icon: "orbit", label: "Exp" },
    // { key: "cosmos", icon: "orbit", label: "Force graph (Cosmos GL)" },
    { key: "table", icon: "view_column", label: "Neighbourhood Table" },
    { key: "diachronic", icon: "calendar_view_week", label: "Diachronic Chart" },
  ] as const;

  return (
    <>
      <nav
        id="app-nav"
        class={`surface-container-low left no-margin top-padding scroll small-elevate ${ open() ? "max" : "small" }`}
      >
        <header class="center-align top-margin tiny-margin no-padding">
          <button
            class="extra transparent no-padding no-margin"
            onClick={() => setOpen(!open())}
          >
            <Icon />
            {/* <Switch>
              <Match when={open()}>
                {" "}
                <i>menu_open</i>
              </Match>
              <Match when={!open()}>
                {" "}
                <i>menu</i>{" "}
              </Match>
            </Switch> */}
          </button>
        </header>

        {navItems.map((item) => (
          <a
            onClick={() => setView(item.key)}
            classList={{
              active: view() === item.key,
              button: true,
              transparent: true,
              "no-border": true,
              "no-padding": true,
              "no-margin": true,
              "no-space": true,
            }}
          >
            <i>{item.icon}</i>
            <span>{item.label}</span>
          </a>
        ))}

        <hr class="max surface-container-low" />

        <a onClick={() => setOpenHelp(!openHelp())} class="extra-padding bottom-padding">
          <i>help</i>
          <span>Guide</span>
        </a>
      </nav>

      <main class="responsive max no-padding full">
        {/* <ErrorBoundary
          fallback={(err, reset) => {
            <AppError err={err as Error} reset={reset} />;
          }}
        > */}
        <Show
          when={dbReady()}
          fallback={
            <article
              class="max small-round padding border medium no-padding"
              style="min-height:100vh"
            >
              <section class="padding absolute center middle">
                <h4>{dbLoadingError() ?? "Loading text events..."}</h4>
                <Show when={!dbLoadingError()}>
                  <progress class="circle" />
                </Show>
              </section>
            </article>
          }
        >
          <Switch>
            <Match when={view() === "table"}>
              <NeighbourhoodBrowser />
            </Match>

            <Match when={view() === "diachronic"}>
              <DiachronicChart />
            </Match>

            {/* <Match when={view() === "cosmos"}>
              <Cosmos />
            </Match> */}

            <Match when={view() === "graph2"}>
              <Graph2 />
            </Match>
          </Switch>
        </Show>
        {/* </ErrorBoundary> */}
      </main>

      <Transition name="slide-fade">
        <Show when={openHelp()}>
          <article
            class="helpContainer right surface-container-high padding  high-elevate border"
            style="z-index:999 !important"
          >
            <Switch fallback={<article>To do...</article>}>
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
