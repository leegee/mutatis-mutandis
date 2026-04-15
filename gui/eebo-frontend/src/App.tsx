import 'beercss';
import { createMemo, Show, onMount, onCleanup, Match, Switch } from "solid-js";

import { data, eeboStore, setEeboStore, setNullSelected } from "./stores/Eebo.store";
import DriftChart, { color } from "./components/DriftChart";
import type { NamedSlicePoint, SlicePoint } from "./types";
import { buildSliceView } from "./models/buildSliceView";
import SliceDensityField from './components/SliceDensityField';
import SLICE_RANGES from "./services/SLICES.json";

export default function App() {
  const series = createMemo(() => {
    const d = data();
    if (!d) return;

    const out: Record<string, SlicePoint[]> = {};

    for (const token of Object.keys(d)) {
      out[token] = d[token].slices.map(s => ({
        ...s
      }));
    }

    return out;
  });

  const currentRange = createMemo(() => {
    return SLICE_RANGES[eeboStore.sliceIndex];
  });

  const sliceView = createMemo(() => {
    const dataset = data();
    const token = eeboStore.selected.token;
    const range = currentRange();

    if (!dataset || !token || !range) return;

    const [sliceStart, sliceEnd] = range;

    const tokenData = dataset[token];
    if (!tokenData) return;

    const slice = tokenData.slices.find(
      s => s.slice_start === sliceStart && s.slice_end === sliceEnd
    );

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

    const list = Object.keys(s); // preserves DriftChart ordering
    const i = list.indexOf(current);
    if (i === -1) return;

    const range = currentRange();
    if (!range) return;

    const [sliceStart, sliceEnd] = range;
    const dataset = data();
    if (!dataset) return;

    // scan to handle missing slices
    for (let j = i + dir; j >= 0 && j < list.length; j += dir) {
      const nextToken = list[j];
      const tokenData = dataset[nextToken];
      if (!tokenData) continue;

      const slice = tokenData.slices.find(
        s => s.slice_start === sliceStart && s.slice_end === sliceEnd
      );

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
        case "Escape":
          setNullSelected();
          break;

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
        <div class="s8 surface-container padding">
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
                    Embeddings created from all EEBO documents in the range 1625-1651 that has a token count in the range of 200-20,000,
                    and whose title does not contain <code>tragedy|comedy|farce|interlude|play</code> and is not obvviously written in Latin.
                  </p>
                  <h2>Usage</h2>
                  <p>
                    Select a term via the checkboxes, which when single-clicked toggle the display of indivual terms,
                    and when double-click show only that term.
                  </p>
                  <p>
                    Once a term is selected, navigate time and terms using the cursor keys.
                  </p>
                  <p>
                    Reset the view with <kbd>ESC</kbd>
                  </p>
                </article>
              </Match>
            </Switch>
          </aside>
        </div>
      </div>
    </main >
  );
}