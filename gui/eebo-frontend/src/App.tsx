import 'beercss';
import { createMemo, onMount, onCleanup, Match, Switch } from "solid-js";

import { data, eeboStore, setEeboStore, setNullSelected } from "./stores/Eebo.store";
import DriftChart, { color } from "./components/DriftChart";
import type { Dataset, NamedSlicePoint, TokenData } from "./types";
import { buildSliceView } from "./models/buildSliceView";
import SliceDensityField from './components/SliceDensityField';
import SLICE_RANGES from "./services/SLICES.json";


export default function App() {

  const series = createMemo(() => {
    let d;
    try {
      d = data();
      if (!d) return;
    } catch (e) {
      console.log('Caught', e);
      return;
    }

    const out: Record<string, Record<string, any>> = {};

    for (const token of Object.keys(d)) {
      out[token] = d[token] ?? {};
    }

    return out;
  });


  const currentRange = createMemo(() => {
    return SLICE_RANGES[eeboStore.sliceIndex];
  });


  const sliceView = createMemo(() => {
    const dataset: Dataset | undefined = data();
    const token = eeboStore.selected.token;
    const range = currentRange();

    if (!dataset || !token || !range) return;

    const [sliceStart, sliceEnd] = range;

    const tokenData = dataset[token];
    if (!tokenData) return;

    const sliceKey = `${sliceStart}-${sliceEnd}`;
    const slice = tokenData[sliceKey as keyof TokenData];

    if (!slice) return;

    return buildSliceView(token, slice, dataset);
  });


  const syncIndexToSlice = (d: NamedSlicePoint) => {
    const idx = SLICE_RANGES.findIndex(
      ([a, b]) => d.slice_start === a && d.slice_end === b
    );

    if (idx >= 0) setEeboStore("sliceIndex", _ => idx);
  };


  const onSelectSlice = (d: NamedSlicePoint) => {
    setEeboStore("selected", {
      token: d.term,
      slice_start: d.slice_start,
      slice_end: d.slice_end,
      color: color(d.term) as string,
    });

    syncIndexToSlice(d);
  };


  const step = (dir: -1 | 1) => {
    setEeboStore("sliceIndex", (i) =>
      Math.max(0, Math.min(
        SLICE_RANGES.length - 1,
        i + dir
      ))
    );
  };


  const stepTerm = (dir: -1 | 1) => {
    const s = series();
    const current = eeboStore.selected.token;

    if (!s || !current) return;

    const list = Object.keys(s);
    const i = list.indexOf(current);
    if (i === -1) return;

    const range = currentRange();
    if (!range) return;

    const [sliceStart, sliceEnd] = range;
    const key = `${sliceStart}-${sliceEnd}`;

    const dataset = data();
    if (!dataset) return;

    for (let j = i + dir; j >= 0 && j < list.length; j += dir) {
      const nextToken = list[j];
      const tokenData = dataset[nextToken];

      if (!tokenData) continue;

      const slice = tokenData[key as keyof TokenData];
      if (!slice) continue;

      setEeboStore("selected", {
        token: nextToken,
        slice_start: sliceStart,
        slice_end: sliceEnd,
        color: color(nextToken) as string,
      });

      break;
    }
  };


  onMount(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.repeat) return;

      if (e.key === "Escape") {
        setNullSelected();
        return;
      }

      if (!eeboStore.selected.token) return;

      switch (e.key) {
        case "ArrowRight":
          step(1);
          e.preventDefault();
          break;

        case "ArrowLeft":
          step(-1);
          e.preventDefault();
          break;

        case "ArrowUp":
          stepTerm(-1);
          e.preventDefault();
          break;

        case "ArrowDown":
          stepTerm(1);
          e.preventDefault();
          break;
      }
    };

    window.addEventListener("keydown", handler);

    onCleanup(() => {
      window.removeEventListener("keydown", handler);
    });
  });


  return (
    <main class="responsive max large-gap">
      <div class="grid" style={{ height: '100%' }}>

        <div class="s8 surface-container">
          <DriftChart
            series={series()!}
            onSelectSlice={(d) => onSelectSlice(d)}
          />
        </div>

        <div class="s4">
          <aside class='surface-container-low center-align middle-align' style={{ height: '100%' }}>

            <Switch>
              <Match when={sliceView()}>
                <SliceDensityField slice={sliceView()!} />
              </Match>

              <Match when={!sliceView()}>
                <article class='border padding large-elevate'>
                  <h1>EEBO Pamphlets</h1>

                  <p>
                    Embeddings from EEBO pamphlets filtered by token count and genre constraints.
                  </p>

                  <p>
                    Horiztonal axis is chronological slice of the corpus.
                  </p>
                  <p>
                    Vertical access is Jensen–Shannon divergence between consecutive slices for a given token.
                  </p>

                  <h2>Usage</h2>

                  <p>
                    Select a point, then navigate time and terms using cursor keys.
                  </p>

                  <p>
                    Filter terms via legend controls (double-click isolates term).
                  </p>

                  <p>
                    Reset view with <kbd>ESC</kbd>
                  </p>
                </article>
              </Match>
            </Switch>

          </aside>
        </div>

      </div>
    </main>
  );
}
