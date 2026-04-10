import 'beercss';
import { createMemo, } from "solid-js";
import { data, eeboStore } from "./stores/Eebo.store";

import NeighborGraph from "./components/NeighborGraph";
import DriftChart from "./components/DriftChart";
import type { SlicePoint } from "./types"

import { buildSliceView } from "./models/buildSliceView";

export default function App() {
  const overlay = () => eeboStore.overlay;

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


  return (
    <main style={{ position: "relative" }}>

      <article style={{ width: "100%", height: "90%" }}>
        <DriftChart
          series={series()!}
          onSelectSlice={
            () => { }
            // (term: string, slice_start: number) => {
            // const dataset = data();
            // if (!dataset || !eeboStore.selected.token) return;
            // const slice = dataset[eeboStore.selected.token].slices.find(
            //   s => s.slice_start === Number(slice_start)
            // );
            // if (!slice) return;
            // console.log('[App] selected in store:', term, "=", eeboStore.selected.token, slice)
            // }
          }
        />
      </article>

      {overlay().open && sliceView() && (
        <div
          style={{
            position: "absolute",
            left: `${overlay().x}px`,
            top: `${overlay().y}px`,
            transform: "translate(-50%, -50%)",
            "pointer-events": "none"
          }}
        >
          <NeighborGraph
            slice={sliceView()!}
            width={1000}
            height={1000}
          />
        </div>
      )}

    </main>
  );
}
