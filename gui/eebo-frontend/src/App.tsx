import { createSignal, Show } from "solid-js";
import { useLocation } from "@solidjs/router";
import { Transition } from "solid-transition-group";

import "./App.css";
import { dbError, dbReady, loadTier2Data } from "./state/tier2data.store";
import GlobalMessageDisplay from "./components/GlobalMessageDisplay";
import AppNav from "./components/AppNav";
import { routes } from "./routes";
import { matchRoute } from "./lib/matchRoute";
import { openHelp, setOpenHelp } from "./state/help.store";
import { GuidePanel } from "./Layout/GuidePanel";
import { ToastHost } from "./components/ToastHost";

export default function App(props: any) {
  const location = useLocation();
  const currentRoute = () => routes.find(r => matchRoute(r.path, location.pathname));
  const [dbLoadingError, setDbLoadingError] = createSignal<string | null>(null);


  console.log("[app] now calling loadTier2Data")

  loadTier2Data()
    .then(() => console.log("[app] loadTier2Data call resolved"))
    .catch(
      (e) => {
        console.warn('[app] caught from loadTier2Data:' + e.message)
        setDbLoadingError(e.message)
      }
    )
    ;


  return (
    <>
      <ToastHost />
      <AppNav />

      <main class="responsive max no-padding full">
        <Show when={dbReady()}
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

      <Transition name="slide-fade">
        <Show when={openHelp() && currentRoute()?.help}>
          <GuidePanel currentRoute={currentRoute} onClose={() => setOpenHelp(false)} />
        </Show>
      </Transition>
    </>
  );
}
