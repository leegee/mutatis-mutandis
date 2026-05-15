import 'beercss';
import { createEffect, createSignal, onMount, Show } from "solid-js";

import Tier3Graph from './components/Tier3Graph';
import { loadDriftData } from './services/zarrJsonService';
import type { Tier3GraphData } from './types';

export default function App() {
  const [data, setData] = createSignal<Tier3GraphData | null>(null);
  const [error, setError] = createSignal<string | null>(null);

  createEffect(() => {
    if (error() !== null) console.error(error()) // later: toast
  })

  onMount(async () => {
    try {
      const result = await loadDriftData("d3_export.json");
      setData(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    }
  });

  return (
    <main class="responsive max large-gap">

      <Show when={!error()} fallback={<p>{error()}</p>}>
        <Show when={data()} fallback={<p>Loading drift data...</p>}>
          {(d) => (
            <Tier3Graph data={d()} />
          )}
        </Show>
      </Show>

    </main>
  );
}
