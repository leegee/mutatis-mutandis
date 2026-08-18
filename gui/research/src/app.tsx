import { MetaProvider, Title } from "@solidjs/meta";
import { Router, useLocation } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense } from "solid-js";
import "beercss/dist/cdn/beer.min.css";

import "./app.css";

function Navigation() {
  const location = useLocation();

  return (
    <nav class="left-padding right-padding fill no-margin">
      <a classList={{ active: location.pathname === "/" }} href="/">
        Map
      </a>
      <a classList={{ active: location.pathname === "/entities" }} href="/entities">
        Entities
      </a>
      <a classList={{ active: location.pathname === "/relations" }} href="/relations">
        Relations
      </a>
      <a classList={{ active: location.pathname === "/project" }} href="/project">
        Project
      </a>
    </nav>
  );
}

export default function App() {
  return (
    <Router
      root={props => (
        <MetaProvider>
          <Title>Research</Title>
          <header>
            <Navigation />
          </header>

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
