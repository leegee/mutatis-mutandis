import 'beercss';
import { createMemo, Show, onMount, onCleanup } from "solid-js";
import {
  closeOverlay,
  data,
  eeboStore,
  openOverlay,
  OVERLAY_SIZE,
  setEeboStore
} from "./stores/Eebo.store";

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
        slice_start: s.slice_start,
        slice_end: s.slice_end,
        drift: s.drift
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

  const onSelectSlice = (d: NamedSlicePoint, x?: number, y?: number) => {
    x = x ?? (window.innerWidth / 2 - OVERLAY_SIZE.width / 2);
    y = y ?? (window.innerHeight / 2 - OVERLAY_SIZE.height / 2);
    setEeboStore("selected", {
      token: d.term,
      slice_start: d.slice_start,
      slice_end: d.slice_end,
      color: color(d.term) as string,
    });

    syncIndexToSlice(d);
    openOverlay(x, y);
  };

  const step = (dir: -1 | 1) => {
    setEeboStore("sliceIndex", (i) =>
      Math.max(0, Math.min(
        SLICE_RANGES.length - 1,
        i + dir
      ))
    );
  };

  onMount(() => {
    const handler = (e: KeyboardEvent) => {
      if (!eeboStore._overlay.open) return;
      if (e.repeat) return;

      if (e.key === "ArrowRight") {
        step(1);
        e.preventDefault();
      }

      if (e.key === "ArrowLeft") {
        step(-1);
        e.preventDefault();
      }
    };

    window.addEventListener("keydown", handler);

    onCleanup(() => {
      window.removeEventListener("keydown", handler);
    });
  });

  return (
    <main class="responsive max">
      <div class="grid" style={{ height: '80%' }}>
        <div class="s8">
          <DriftChart
            series={series()!}
            onSelectSlice={(d, x, y) => onSelectSlice(d, x, y)}
          />
        </div>

        <div class="s4">
          <Show when={eeboStore._overlay.open && sliceView()}>
            <aside class='surface-container' style={{ height: '100%' }}>
              <SliceDensityField slice={sliceView()!} />
            </aside>
          </Show>

        </div>
      </div>
    </main >
  );
}