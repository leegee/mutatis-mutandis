import { createEffect, createMemo } from "solid-js";
import * as d3 from "d3";

import { setEeboStore, data, eeboStore } from "./stores/Eebo.store";

import NeighborGraph from "./components/NeighborGraph";
import DriftChart from "./components/DriftChart";

import type { Slice, SliceView } from "./types";

export default function App() {
  const store = eeboStore;

  console.log("[App] Init with store:", store);

  // -----------------------------
  // Slice (current selection)
  // -----------------------------
  const slice = createMemo<Slice | undefined>(() => {
    const dataset = data();

    const token = store.selected.token;
    const slice_start = store.selected.slice_start;

    console.info("[App] deps", {
      token,
      slice_start,
      hasData: !!dataset
    });

    if (!dataset) return;

    if (!token || slice_start == null || !dataset[token]) {
      console.log("[App] BAIL", {
        token,
        slice_start
      });
      return;
    }

    const tokenData = dataset[token];

    return tokenData.slices.find(
      (s: Slice) => s.slice_start === slice_start
    );
  });

  // -----------------------------
  // SliceView (THE VIEW MODEL)
  // -----------------------------
  const sliceView = createMemo<SliceView | undefined>(() => {
    const dataset = data();
    const s = slice();
    const sel = store.selected;

    if (!dataset || !s || !sel.token) return;

    const tokenData = dataset[sel.token];

    // -----------------------------
    // Neighbors + rank
    // -----------------------------
    const neighbors = s.top_neighbors ?? [];

    const rank = new Map(
      neighbors.map((n, i) => [n.token, i])
    );

    // -----------------------------
    // 🔥 HISTORY (time series)
    // -----------------------------
    const raw = tokenData.slices.map(sl => ({
      t: sl.slice_start,
      drift: sl.drift ?? 0
    }));

    // first derivative (velocity)
    const withD1 = raw.map((d, i) => ({
      ...d,
      d1: i === 0 ? 0 : d.drift - raw[i - 1].drift
    }));

    // second derivative (acceleration / shocks)
    const history = withD1.map((d, i) => ({
      ...d,
      d2: i < 2 ? 0 : d.d1 - withD1[i - 1].d1
    }));

    // -----------------------------
    // 🔥 TRANSITIONS (d2 spikes)
    // -----------------------------
    const d2Values = history.map(d => d.d2);

    const threshold =
      (d3.deviation(d2Values) ?? 0) * 2;

    const transitions = history
      .filter(d => Math.abs(d.d2) > threshold)
      .map(d => d.t);

    console.log("[App] sliceView built", {
      token: sel.token,
      slices: history.length,
      transitions: transitions.length
    });

    return {
      token: sel.token,
      slice_start: s.slice_start,
      slice_end: s.slice_end ?? s.slice_start,

      neighbors,
      drift: s.drift ?? 0,
      normalizedDrift: Math.min(1, (s.drift ?? 0) / 5),

      rank,

      // 🔥 new signal layer
      history,
      transitions
    };
  });

  // -----------------------------
  // Bootstrap selection
  // -----------------------------
  createEffect(() => {
    const d = data();
    if (!d) return;

    const sel = store.selected;
    if (sel.token && sel.slice_start != null) return;

    const firstToken = Object.keys(d).sort()[0];
    const firstSlice = d[firstToken].slices[0];

    console.log("[App] Bootstrapping selection", firstSlice);

    const slice_start = firstSlice.slice_start;

    if (slice_start == null) {
      console.error("[App] Cannot determine slice_start", firstSlice);
      return;
    }

    setEeboStore("selected", {
      token: firstToken,
      slice_start,
      slice_end: firstSlice.slice_end ?? slice_start,
      color: "#000"
    });
  });

  return (
    <main>
      {sliceView() && (
        <>
          <DriftChart slice={sliceView()} />
          <NeighborGraph slice={sliceView()} />
        </>
      )}
    </main>
  );
}
