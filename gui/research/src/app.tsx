import { MetaProvider, Title } from "@solidjs/meta";
import { Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense } from "solid-js";
import "beercss/dist/cdn/beer.min.css";

import "./app.css";

export default function App() {
  return (
    <Router
      root={props => (
        <MetaProvider>
          <Title>Research</Title>
          <header>
            <nav class="large-padding left-padding right-padding fill no-margin">
              <a href="/">Home</a>
              <a href="/entities">Entities</a>
              <a href="/relations">Relations</a>
              <a href="/project">Project</a>
            </nav>
          </header>

          <main class="responsive max large-padding">
            <Suspense>{props.children}</Suspense>
          </main>
        </MetaProvider>
      )}
    >
      <FileRoutes />
    </Router>
  );
}
