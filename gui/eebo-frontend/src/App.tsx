import { createSignal, Show } from "solid-js";
import { useLocation } from "@solidjs/router";
import { Transition } from "solid-transition-group";

import "./App.css";
import { dbReady, loadTier2Data } from "./state/tier2data.store";
import GlobalMessageDisplay from "./components/GlobalMessageDisplay";
import AppNav from "./components/AppNav";
import { routes } from "./routes";
import { openHelp } from "./state/help";

export default function App(props: any) {
  const location = useLocation();

  const [dbLoadingError, setDbLoadingError] = createSignal<string | null>(null);

  const matchRoute = (pattern: string, path: string) => {
    const clean = (p: string) => p.split("?")[0]; // ignore optional marker

    const pParts = clean(pattern).split("/").filter(Boolean);
    const pathParts = path.split("/").filter(Boolean);

    if (pParts.length !== pathParts.length) return false;

    return pParts.every((p, i) => {
      if (p.startsWith(":")) return true;
      return p === pathParts[i];
    });
  };

  const currentRoute = () =>
    routes.find(r => matchRoute(r.path, location.pathname));

  try {
    loadTier2Data();
  } catch (e) {
    setDbLoadingError((e as Error).message);
  }

  const HelpPanel = () => {
    const route = currentRoute();
    if (!route?.help) return null;

    return (
      <article
        class="helpContainer right surface-container-high padding high-elevate border"
        style="z-index:999"
      >
        {route.help()}
      </article>
    );
  };

  return (
    <>
      <AppNav />

      <main class="responsive max no-padding full">
        <Show
          when={dbReady()}
          fallback={
            <GlobalMessageDisplay
              title="Loading database"
              errorMessage={dbLoadingError()}
            />
          }
        >
          {props.children}
        </Show>
      </main>

      {/* Help toggle button is assumed to live in AppNav or elsewhere */}

      <Transition name="slide-fade">
        <Show when={openHelp() && currentRoute()?.help}>
          <HelpPanel />
        </Show>
      </Transition>
    </>
  );
}
