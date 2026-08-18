import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense, createSignal } from "solid-js";
import "beercss/dist/cdn/beer.min.css";

import "./app.css";
import "./nav-menu.css";

function Navigation() {
  const location = useLocation();
  const [minimized, setMinimized] = createSignal(false);

  return (
    <nav class={`left surface-container ${ minimized() ? "minimized" : "maximized" }`}>
      <header>
        <nav>
          <button
            class="transparent circle"
            onClick={() => setMinimized(!minimized())}
            aria-label={minimized() ? "Expand menu" : "Minimize menu"}
            title={minimized() ? "Expand menu" : "Minimize menu"}
          >
            <i>{minimized() ? "menu" : "menu_open"}</i>
          </button>

          <span>Foo Map</span>
        </nav>
      </header>

      <a
        href="/"
        aria-label="Map"
        aria-current={location.pathname === "/" ? "page" : undefined}
        classList={{ active: location.pathname === "/" }}
      >
        <i>graph_7</i>
        <span>Map</span>
      </a>

      <a
        href="/entities"
        aria-label="Entities"
        aria-current={location.pathname === "/entities" ? "page" : undefined}
        classList={{ active: location.pathname === "/entities" }}
      >
        <i>add_circle</i>
        <span>Entities</span>
      </a>

      <a
        href="/relations"
        aria-label="Relations"
        aria-current={location.pathname === "/relations" ? "page" : undefined}
        classList={{ active: location.pathname === "/relations" }}
      >
        <i>arrow_and_edge</i>
        <span>Relations</span>
      </a>

      <a
        href="/project"
        aria-label="Project"
        aria-current={location.pathname === "/project" ? "page" : undefined}
        classList={{ active: location.pathname === "/project" }}
      >
        <i>folder_open</i>
        <span>Project</span>
      </a>

      <div class="divider"></div>
    </nav>
  );
}

export default function App() {
  return (
    <Router
      root={props => (
        <MetaProvider>
          <Title>Research</Title>
          <Navigation />

          <main class="responsive max no-padding">
            <Suspense>{props.children}</Suspense>
          </main>
        </MetaProvider>
      )}
    >
      <FileRoutes />
    </Router>
  );
}
