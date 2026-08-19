import { useLocation } from "@solidjs/router";
import { createSignal } from "solid-js";

export function SideNavigation() {
  const location = useLocation();
  const [minimized, setMinimized] = createSignal(false);

  return (
    <nav
      class={`left surface-container ${ minimized() ? "minimized" : "maximized" }`}
    >
      <header>
        <nav>
          <button type="button"
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
        <span>Import/Export</span>
      </a>

      <div class="divider"></div>
    </nav>
  );
}
