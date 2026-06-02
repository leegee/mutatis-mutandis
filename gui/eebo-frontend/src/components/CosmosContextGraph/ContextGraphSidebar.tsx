// ContextGraphSidebar.tsx

import { Show, For, type Component } from "solid-js";

import type { ContextGraphData, ContextNode, TokenBin } from "./types";

import { showDocument } from "../../services/documentApi";
import EventContext from "../EventContext";

interface SharedHub {
  hub: string;
  freq: number;
  meanScore: number;
}

interface Props {
  viewMode: string;
  selectedNode: string;
  selectedKind: "hub" | "event" | "neighbour" | null;
  selectedBin: TokenBin | null;
  selectedDocs: Array<[string, number | undefined]>;
  selectedEventNode: ContextNode | null;
  sharedByHubs: SharedHub[];
  graphData: ContextGraphData;
  onClose: () => void;
  maxTopN: number;
}

const ContextGraphSidebar: Component<Props> = (props) => {
  return (
    <aside
      class="min surface-container-high scroll medium-elevate padding border small-margin no-top-padding"
      style="max-width: 20vw; min-width: 20rem"
    >
      <div class="cg-header-row">
        <h2>
          <q>{props.selectedNode}</q>
        </h2>

        <button class="link border" onClick={props.onClose}>
          ✕
        </button>
      </div>

      <Show when={props.selectedKind === "hub" && props.selectedBin}>
        {(_) => {
          const bin = props.selectedBin!;
          const years = [...bin.years].sort((a, b) => a - b);
          const topMax = bin.topNeighbours[0]?.freq ?? 1;
          return (
            <>
              <div class="bottom-padding">
                <div>Events: {bin.eventCount}</div>
                <div>Documents: {bin.docs.size}</div>
                <div>
                  Years:{" "}
                  {years.length
                    ? years.length === 1
                      ? years[0]
                      : `${years[0]}–${years[years.length - 1]}`
                    : "-"}
                </div>
                <div>
                  Hub connections:{" "}
                  {props.graphData.nodes.find(
                    (n) => n.id === props.selectedNode,
                  )?.hubDegree ?? 0}
                </div>
              </div>

              <details open={true}>
                <summary>
                  <h3 class="bottom-padding">Top neighbours</h3>
                </summary>
                <div class="bottom-padding">
                  <For each={bin.topNeighbours.slice(0, props.maxTopN)}>
                    {(nb) => (
                      <div class="row max">
                        <div class="cg-nb-bar-wrap" style="width: 33%">
                          <div
                            class="cg-nb-bar-fill hub"
                            style={{
                              width: `${(nb.freq / topMax) * 100}%`,
                            }}
                          />
                        </div>
                        <span class="cg-nb-token">
                          <q>{nb.token}</q>
                        </span>
                        <span class="cg-nb-score">
                          {nb.meanScore.toFixed(3)}
                        </span>
                      </div>
                    )}
                  </For>
                </div>
              </details>

              <details open={true}>
                <summary>
                  <h3 class="bottom-padding">Sources</h3>
                </summary>
                <Show
                  when={props.selectedDocs.length > 0}
                  fallback={<div class="error">No documents found</div>}
                >
                  <For each={props.selectedDocs}>
                    {([docId, pubYear]) => (
                      <button
                        class="chip small-margin cg-chip-mono"
                        onClick={() => showDocument(docId)}
                      >
                        <span>{docId}</span>
                        <span class="small-text"> {pubYear}</span>
                      </button>
                    )}
                  </For>
                </Show>
              </details>
            </>
          );
        }}
      </Show>

      <Show when={props.selectedKind === "event" && props.selectedEventNode}>
        {(_) => {
          const node = props.selectedEventNode!;
          return (
            <>
              <div class="bottom-padding">
                <div>
                  Token: <q>{node.token ?? "-"}</q>
                </div>
                <div>Year: {node.pub_year ?? "-"}</div>
                <div>
                  Document: {node.doc_id} token {node.token_idx ?? "-"}
                  <EventContext
                    docId={node.doc_id!}
                    tokenIdx={node.token_idx!}
                    open={true}
                  />
                </div>
                <Show when={node.doc_id}>
                  <div>
                    <button
                      class="chip small-margin cg-chip-mono"
                      onClick={() => showDocument(node.doc_id!)}
                    >
                      <span>{node.doc_id}</span>
                    </button>
                  </div>
                </Show>
              </div>
              <div class="bottom-padding small-text" style={{ opacity: 0.6 }}>
                Select a neighbour to see which sources share it.
              </div>
            </>
          );
        }}
      </Show>

      <Show when={props.selectedKind === "neighbour"}>
        <div class="bottom-padding">
          <div>Shared by {props.sharedByHubs.length} source(s)</div>
        </div>
        <h3 class="bottom-padding">
          {props.viewMode === "aggregated" ? "Hub contexts" : "Event contexts"}
        </h3>
        <Show
          when={props.sharedByHubs.length > 0}
          fallback={<div class="error">Not in any top-N list</div>}
        >
          {(_) => {
            const maxFreq = props.sharedByHubs[0]?.freq ?? 1;
            return (
              <div class="bottom-padding">
                <For each={props.sharedByHubs}>
                  {(h) => {
                    const sourceNode = () =>
                      props.viewMode === "events"
                        ? props.graphData.nodes.find((n) => n.id === h.hub)
                        : null;
                    return (
                      <>
                        <article>
                          <div class="row">
                            <div class="cg-nb-bar-wrap">
                              <div
                                class="cg-nb-bar-fill neighbour"
                                style={{
                                  width: `${(h.freq / maxFreq) * 100}%`,
                                }}
                              />
                            </div>
                            <span class="tooltip bottom">
                              Mean score: {h.meanScore.toFixed(3)}
                            </span>
                          </div>

                          <Show when={sourceNode()}>
                            {(neighbourToken) => (
                              <>
                                <span>{sourceNode()?.token ?? h.hub}</span>{" "}
                                <span
                                  class="small-text"
                                  style={{ opacity: 0.6 }}
                                >
                                  {neighbourToken().doc_id}/
                                  {neighbourToken().token_idx}
                                </span>
                                <EventContext
                                  open={true}
                                  docId={neighbourToken().doc_id!}
                                  tokenIdx={neighbourToken().token_idx!}
                                />
                              </>
                            )}
                          </Show>
                        </article>
                      </>
                    );
                  }}
                </For>
              </div>
            );
          }}
        </Show>
      </Show>
    </aside>
  );
};

export default ContextGraphSidebar;
