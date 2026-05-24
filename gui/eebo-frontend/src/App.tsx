// src/App.tsx

import { createResource, ErrorBoundary, Show } from "solid-js";

import ConceptGraph from "./components/ConceptGraph";
import { loadConceptNeighbours } from "./services/loadConceptNeighbours";

export default function App() {
  const [events] = createResource(loadConceptNeighbours);

  return (
    <main class="responsive max">
      <ErrorBoundary fallback={(err) => <article><div class="error padding">{err.message}</div></article>}>
        <Show when={events()} fallback="Loading events...">
          {(data) => <ConceptGraph data={data()} />}
        </Show>
      </ErrorBoundary>
    </main>
  );
}
