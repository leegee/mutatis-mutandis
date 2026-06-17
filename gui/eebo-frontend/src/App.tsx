import { createSignal, Show, lazy, ErrorBoundary } from "solid-js";
import { useLocation } from "@solidjs/router";
import { Transition } from "solid-transition-group";

import "./App.css";
import { dbReady, loadTier2Data } from "./state/tier2data.store";
import AppError from "./components/AppError";
import GlobalMessageDisplay from "./components/GlobalMessageDisplay";
import AppNav from "./components/AppNav";

export default function App(props: any) {
  const location = useLocation();

  const [dbLoadingError, setDbLoadingError] = createSignal<string | null>(null);
  const [openHelp, setOpenHelp] = createSignal(false);

  try {
    loadTier2Data();
  } catch (e) {
    setDbLoadingError((e as Error).message);
  }

  return (
    <>
      <AppNav />
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

      {/* <Transition name="slide-fade">
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
      </Transition> */}
    </>
  );
}
