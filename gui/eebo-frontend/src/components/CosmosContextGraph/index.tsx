import { createMemo, createResource, Show, type Component } from "solid-js";

import "./styles.css";

import type { ContextGraphData, ContextNode, TokenBin } from "./types";

import { controls, setControls } from "../../state/controls.store";
import {
  aggregateByToken,
  buildContextualGraph,
  buildPureEventGraph,
} from "../../lib/contextGraphUtils";
import {
  getYearFiltered,
  getYearBounds,
  totalEventsForConcept,
} from "../../state/selectors";
import ControlsHeader from "../ControlsHeader";
import ContextGraphSidebar from "./ContextGraphSidebar";
import GraphCanvas, { type GraphFrame } from "./GraphCanvas";
import { controlsActions } from "../../state/controls.actions";

const MAX_TOP_N = 20;

const EMPTY_GRAPH: ContextGraphData = {
  nodes: [],
  hubHubEdges: [],
  hubNbEdges: [],
  allEdges: [],
  maxHubHubWeight: 1,
  maxEventCount: 1,
  maxHubDegree: 1,
};

const CosmosComponent: Component = () => {
  const [filteredEventsResource] = createResource(
    () => [controls.concept, controls.fromYear, controls.toYear] as const,
    ([concept, from, to]) => getYearFiltered(concept, from, to),
  );
  const filteredEvents = () => filteredEventsResource() ?? [];

  const [yearBoundsResource] = createResource(
    () => controls.concept,
    (concept) => getYearBounds(concept),
  );
  const yearBounds = (): [number, number] =>
    yearBoundsResource() ?? [controls.fromYear, controls.toYear];

  const [totalEventsResource] = createResource(
    () => controls.concept,
    (concept) => totalEventsForConcept(concept),
  );
  const totalEvents = () => totalEventsResource() ?? 0;

  const tokenBins = createMemo<Map<string, TokenBin>>(() =>
    aggregateByToken(filteredEvents()),
  );

  const graphData = createMemo<ContextGraphData>(() =>
    controls.viewMode === "events"
      ? buildPureEventGraph(filteredEvents(), controls.topN, EMPTY_GRAPH)
      : buildContextualGraph(
        tokenBins(),
        controls.topN,
        controls.minSimilarity,
        controls.maxHubs,
        EMPTY_GRAPH,
      ),
  );

  const selectedKind = createMemo<"hub" | "neighbour" | "event" | null>(() => {
    const id = controls.selectedNode;
    if (!id) return null;
    return graphData().nodes.find((n) => n.id === id)?.kind ?? null;
  });

  const selectedBin = createMemo<TokenBin | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "hub") return null;
    return tokenBins().get(id) ?? null;
  });

  const selectedDocs = createMemo<Array<[string, number | undefined]>>(() => {
    const bin = selectedBin();
    if (!bin) return [];
    return [...bin.docs.entries()].sort(
      (a, b) => (a[1] ?? Infinity) - (b[1] ?? Infinity),
    );
  });

  const selectedEventNode = createMemo<ContextNode | null>(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "event") return null;
    return graphData().nodes.find((n) => n.id === id) ?? null;
  });

  const sharedByHubs = createMemo<
    Array<{ hub: string; freq: number; meanScore: number }>
  >(() => {
    const id = controls.selectedNode;
    if (!id || selectedKind() !== "neighbour") return [];
    if (controls.viewMode === "aggregated") {
      const result: Array<{ hub: string; freq: number; meanScore: number }> =
        [];
      for (const [hubKey, bin] of tokenBins()) {
        const nb = bin.topNeighbours.find((n) => n.token === id);
        if (nb)
          result.push({ hub: hubKey, freq: nb.freq, meanScore: nb.meanScore });
      }
      return result.sort((a, b) => b.freq - a.freq);
    }
    return graphData()
      .hubNbEdges.filter((e) => e.targetId === id)
      .map((e) => ({ hub: e.sourceId, freq: e.weight, meanScore: e.weight }))
      .sort((a, b) => b.freq - a.freq);
  });

  const graphCanvasFrame = createMemo<GraphFrame>(() => ({
    graphData: graphData(),
    tokenBins: tokenBins(),
    concept: controls.concept,
    viewMode: controls.viewMode,
    hubSpread: controls.hubSpread,
    showEventLabels: controls.showEventLabels,
    toYear: controls.toYear,
    fromYear: controls.fromYear,
    selectedNode: controls.selectedNode,
    onSelectNode: (id: string | null) => {
      // setControls("selectedNode", (prev) => (prev === id ? null : id));
      controlsActions.setSelectedNode(id);
      console.log("[graph index onSelectNode]", id);
    },
  }));

  return (
    <>
      <div class="background cg-layout">
        <ControlsHeader totalEvents={totalEvents} includeHubSpread={true}>
          {/* <div class="field center-align">
            <button class="small transparent border" onclick={forceRedraw}>
              <i>redo</i>
            </button>
            <output>Redraw</output>
            <span class="tooltip bottom">Update the graph layout</span>
          </div> */}
        </ControlsHeader>

        <div class="cg-main">
          <GraphCanvas frame={graphCanvasFrame()} />

          <Show when={controls.selectedNode}>
            <Show when={controls.selectedNode}>
              <ContextGraphSidebar
                maxTopN={MAX_TOP_N}
                selectedNode={controls.selectedNode!}
                selectedKind={selectedKind()}
                graphData={graphData()}
                selectedBin={selectedBin()}
                selectedDocs={selectedDocs()}
                selectedEventNode={selectedEventNode()}
                sharedByHubs={sharedByHubs()}
                viewMode={controls.viewMode}
                onClose={() => setControls("selectedNode", null)}
              />
            </Show>
          </Show>
        </div>

        <footer class="fixed max center-align small-padding surface-container-low">
          <span class="cg-legend">
            <Show when={controls.viewMode === "aggregated"}>
              <span class="cg-legend-hub" /> hubs (
              {graphData().nodes.filter((n) => n.kind === "hub").length})
            </Show>
            <Show when={controls.viewMode === "events"}>
              <span class="cg-legend-event" /> events (
              {graphData().nodes.filter((n) => n.kind === "event").length})
            </Show>{" "}
            <span class="cg-legend-nb" /> neighbours (
            {graphData().nodes.filter((n) => n.kind === "neighbour").length})
            <Show when={controls.viewMode === "aggregated"}>
              {" • "}
              {graphData().hubHubEdges.length} similarity edges
            </Show>
            {" • "}
            {graphData().hubNbEdges.length} spokes
            {" • "}
            {filteredEvents().length} events
            <Show
              when={
                controls.fromYear !== yearBounds()[0] ||
                controls.toYear !== yearBounds()[1]
              }
            >
              {" • "}
              {controls.fromYear}–{controls.toYear}
            </Show>
            {" • "}
            {totalEvents()} total
          </span>
        </footer>
      </div>
    </>
  );
};

export default CosmosComponent;
