import 'beercss';
import { createMemo, } from "solid-js";
import { data, eeboStore, openOverlay, OVERLAY_SIZE, setEeboStore } from "./stores/Eebo.store";

import NeighborGraph from "./components/NeighborGraph";
import DriftChart, { color } from "./components/DriftChart";
import type { NamedSlicePoint, Slice, SlicePoint } from "./types"

import { buildSliceView } from "./models/buildSliceView";

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

  const sliceView = createMemo(() => {
    const dataset = data();
    const token = eeboStore.selected.token;
    const sliceStart = eeboStore.selected.slice_start;

    if (!dataset || !token || sliceStart == null) return;

    const tokenData = dataset[token];
    const slice = tokenData.slices.find(
      s => s.slice_start === sliceStart
    );

    if (!slice) return;

    return buildSliceView(token, slice, dataset);
  });


  const onSelectSlice = (d: NamedSlicePoint, x: number, y: number) => {
    setEeboStore("selected", {
      token: d.term,
      slice_start: d.slice_start,
      slice_end: d.slice_end,
      color: color(d.term) as string,
    });

    openOverlay(x, y);
  };

  return (
    <main>
      <article style={{ width: "100%", height: "90%" }}>

        <DriftChart series={series()!}
          onSelectSlice={(d, x, y) => onSelectSlice(d, x, y)}
        />

        {eeboStore._overlay.open && sliceView() && (
          <aside
            style={{
              position: "fixed",
              left: `${eeboStore._overlay.x}px`,
              top: `${eeboStore._overlay.y}px`,
              "pointer-events": "none"
            }}
          >
            <NeighborGraph
              slice={sliceView()!}
              width={OVERLAY_SIZE.width}
              height={OVERLAY_SIZE.height}
            />
          </aside>
        )}
      </article>

    </main>
  );
}
