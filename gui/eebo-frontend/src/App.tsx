import { createSignal, Show, lazy, ErrorBoundary } from "solid-js";
import { A, useLocation } from "@solidjs/router";
import { Transition } from "solid-transition-group";

import "./App.css";
import { dbReady, loadTier2Data } from "./state/tier2data.store";
import AppError from "./components/AppError";
import { Icon } from "./components/Icon";
import GlobalMessageDisplay from "./components/GlobalMessageDisplay";

const CosmosContextGraphGuide = lazy(
  () => import("./components/CosmosContextGraph/Guide"),
);

export default function App(props: any) {
  const location = useLocation();

  const [dbLoadingError, setDbLoadingError] = createSignal<string | null>(null);
  const [openHelp, setOpenHelp] = createSignal(false);
  const [open, setOpen] = createSignal(false);

  try {
    loadTier2Data();
  } catch (e) {
    setDbLoadingError((e as Error).message);
  }

  const navItems = [
    { path: "/scatter", icon: "scatter_plot", label: "UMAP" },
    { path: "/aggregates", icon: "crowdsource", label: "Aggregates" },
    { path: "/graph2", icon: "orbit", label: "FDG" },
    { path: "/table", icon: "view_column", label: "Neighbourhood Table" },
    { path: "/diachronic", icon: "calendar_view_week", label: "Diachronic Chart" },
  ] as const;

  const isActive = (path: string) => location.pathname === path;

  return (
    <>
      <nav id="app-nav" class={`surface-container-low left no-margin top-padding scroll small-elevate ${ open() ? "max" : "small" }`} >
        <header class="center-align top-margin tiny-margin no-padding">
          <button
            class="extra transparent no-padding no-margin"
            onClick={() => setOpen(!open())}
          >
            <Icon />
          </button>
        </header>

        {navItems.map((item) => (
          <A
            href={item.path}
            classList={{
              active: isActive(item.path),
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
          </A>
        ))}

        <hr class="max surface-container-low" />

        <a
          onClick={() => setOpenHelp(!openHelp())}
          class="extra-padding bottom-padding"
        >
          <i>help</i>
          <span>Guide</span>
        </a>
      </nav>

      <main class="responsive max no-padding full">
        {/* <ErrorBoundary
          fallback={(err, reset) => (
            <AppError err={err as Error} reset={reset} />
          )}
        > */}
        <Show
          when={dbReady()}
          fallback={
            <GlobalMessageDisplay title="Loading database" errorMessage={dbLoadingError()} />
          }
        >
          {props.children}
        </Show>
        {/* </ErrorBoundary> */}
      </main>

      <Transition name="slide-fade">
        <Show when={openHelp()}>
          <article
            class="helpContainer right surface-container-high padding high-elevate border"
            style="z-index:999 !important"
          >
            <Show when={location.pathname === "/cosmos"}>
              <CosmosContextGraphGuide />
            </Show>
          </article>
        </Show>
      </Transition>
    </>
  );
}
