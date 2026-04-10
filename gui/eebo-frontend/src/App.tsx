import { createMemo, createEffect } from "solid-js";
import { data, eeboStore, setEeboStore } from "./stores/Eebo.store";

import NeighborGraph from "./components/NeighborGraph";
import DriftChart from "./components/DriftChart";
import type { SlicePoint } from "./types"

import { buildSliceView } from "./models/buildSliceView";

export default function App() {
  const store = eeboStore;

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
    const sel = store.selected;

    if (!dataset || !sel.token || sel.slice_start == null) return;

    const tokenData = dataset[sel.token];
    const slice = tokenData.slices.find(
      s => s.slice_start === sel.slice_start
    );

    if (!slice) return;

    return buildSliceView(sel.token, slice, dataset);
  });

  createEffect(() => {
    const d = data();
    if (!d) return;

    const sel = store.selected;
    if (sel.token && sel.slice_start != null) return;

    const firstToken = Object.keys(d).sort()[0];
    const firstSlice = d[firstToken].slices[0];

    setEeboStore("selected", {
      token: firstToken,
      slice_start: firstSlice.slice_start,
      slice_end: firstSlice.slice_end ?? firstSlice.slice_start,
      color: "#000"
    });
  });

  return (
    <main>
      {sliceView() && (
        <>
          <div style={{ width: "100%", height: "66%" }}>
            <DriftChart
              series={series()!}
              onSelectSlice={(t) => {
                const dataset = data();
                const token = store.selected.token;
                if (!dataset || !token) return;

                const slice = dataset[token].slices.find(
                  s => s.slice_start === t
                );

                if (!slice) return;

                setEeboStore("selected", {
                  token,
                  slice_start: slice.slice_start,
                  slice_end: slice.slice_end ?? slice.slice_start,
                  color: "#000"
                });
              }}
            />
          </div>

          <NeighborGraph
            slice={sliceView()!}
            width={500}
            height={400}
          />
        </>
      )}
    </main>
  );
}
