import { createSignal, onMount } from "solid-js";

// import EeboSearch from "./components/EeboSearch";
import DriftChart from "./components/DriftChart";
import NeighborGraph from "./components/NeighborGraph";
import type { Dataset, Selection } from "./types";
import { fetchTokenClusters } from "./services/tokenClustersService";

export default function App() {
  const [data, setData] = createSignal<Dataset>();
  const [hovered, setHovered] = createSignal<Selection | null>(null);
  const [selected, setSelected] = createSignal<Selection>({ token: null, year: null, color: "#222" });

  onMount(async () => {
    const json = await fetchTokenClusters("drift_neighbors_micro_senses_slices.json"); // See python/src/mb_test.py `OUT_PATH`
    setData(json);
  });

  return (
    <>
      <DriftChart
        data={data()}
        hovered={hovered}
        setHovered={setHovered}
        selected={selected}
        setSelected={setSelected}
      />

      <NeighborGraph
        token={selected().token}
        neighbors={selected().token ? data()[selected().token].slices.find(s => s.year === selected().year)?.top_neighbors || [] : []}
        drift={selected().token ? data()[selected().token].slices.find(s => s.year === selected().year)?.drift || 0 : 0}
        color={selected().color}
      />
    </>
  )
}
