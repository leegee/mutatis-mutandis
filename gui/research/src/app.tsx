import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense, createSignal } from "solid-js";
import "beercss/dist/cdn/beer.min.css";

import "./app.css";
import "./nav-menu.css";

function SideNavigation() {
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

          <span>The Foo Mapper</span>
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


function Navigation() {
  const location = useLocation();
  const [menuOpen, setMenuOpen] = createSignal(false);

  const toggleMenu = () => {
    setMenuOpen(open => !open);
  };

  const closeMenu = () => {
    setMenuOpen(false);
  };

  const isActive = (path: string) => location.pathname === path;

  return (
    <div class="navigation-button-menu">
      <button class="transparent  margin" onClick={toggleMenu}>
        <i>{menuOpen() ? "menu_open" : "menu"}</i>
        <span>Navigation</span>
        <i>{menuOpen() ? "arrow_drop_up" : "arrow_drop_down"}</i>
      </button>

      {menuOpen() && (
        <menu class="margin">
          <li classList={{ active: isActive("/") }}>
            <a href="/" onClick={closeMenu}>
              <i>graph_7</i>
              <span>Map</span>
            </a>
          </li>

          <li classList={{ active: isActive("/entities") }}>
            <a href="/entities" onClick={closeMenu}>
              <i>add_circle</i>
              <span>Entities</span>
            </a>
          </li>

          <li classList={{ active: isActive("/relations") }}>
            <a href="/relations" onClick={closeMenu}>
              <i>arrow_and_edge</i>
              <span>Relations</span>
            </a>
          </li>

          <li classList={{ active: isActive("/project") }}>
            <a href="/project" onClick={closeMenu}>
              <i>folder_open</i>
              <span>Project</span>
            </a>
          </li>
        </menu>
      )}
    </div>
  );
}

export default function App() {
  return (
    <Router
      root={props => (
        <MetaProvider>
          <Title>Research</Title>
          {/* <SideNavigation /> */}

          <main class="responsive max no-padding background">
            <Suspense>
              <Navigation />
              {props.children}
            </Suspense>
          </main>
        </MetaProvider>
      )}
    >
      <FileRoutes />
    </Router>
  );
}
