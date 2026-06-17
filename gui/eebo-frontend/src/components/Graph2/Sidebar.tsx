// ConceptGraphSidebar.tsx

import { Show, For, type Component } from "solid-js";
import { showDocument } from "../../services/documentApi";
import EventContextWindowText from "../EventContextWindowText";

interface Props {
  selectedNode: NodeMeta | null;
  graphData: Graph2Data | null;
  onClose: () => void;
}

import "./Sidebar.css";
import { type NodeMeta, type Graph2Data, EDGE_KIND, NODE_KIND } from "../../types/tier2_comos_sqlite";

const ConceptGraphSidebar: Component<Props> = (props) => {
  // Edges where this node is src or tgt, with the other end resolved
  const neighbourEdges = () => {
    const node = props.selectedNode;
    const data = props.graphData;
    if (!node || !data) return [];

    const nodeIdx = data.nodes.indexOf(node);
    if (nodeIdx === -1) return [];

    return data.edges
      .filter(
        (e) =>
          e.kind === EDGE_KIND.SEMANTIC &&
          (e.srcIdx === nodeIdx || e.tgtIdx === nodeIdx),
      )
      .map((e) => {
        const otherIdx = e.srcIdx === nodeIdx ? e.tgtIdx : e.srcIdx;
        return { edge: e, neighbour: data.nodes[otherIdx] };
      })
      .sort((a, b) => b.edge.weight - a.edge.weight);
  };

  return (
    <aside
      class="min surface-container-high scroll medium-elevate border no-padding no-margin no-round"
      style="max-width: 20vw; min-width: 20rem"
    >
      <Show when={props.selectedNode} fallback={
        <div style={{ opacity: 0.4, padding: "1rem" }}>Click a node to inspect it.</div>
      }>
        {(_) => {
          const node = props.selectedNode!;
          return (
            <article>
              <header>
                <div class="row">
                  <h2 class="max large"><q>{node.label}</q></h2>
                  <button class="link border" onClick={props.onClose}><i>close</i></button>
                </div>
              </header>

              {/*  Metadata  */}
              <section class="bottom-padding">
                <div>Kind: {["event", "neighbour", "concept"][node.kind]}</div>
                {node.pubYear && <div>Year: {node.pubYear}</div>}
                {node.docId && <div>Document: {node.docId}</div>}
                {node.windowId != null && <div>Window: {node.windowId}</div>}
                {node.tokenIdx != null && <div>Token Index: {node.tokenIdx}</div>}
                {node.degree != null && <div>Degree: {node.degree}</div>}
                <div style={{ opacity: 0.75, "font-size": "10px", "margin-top": "4px" }}>{node.id}</div>
              </section>

              {/*  EventContext (events and neighbours that have a doc + window)  */}
              <Show when={node.kind !== NODE_KIND.CONCEPT && node.docId && node.windowId != null}>
                <section class="bottom-padding">
                  <EventContextWindowText
                    label="Context"
                    docId={node.docId!}
                    tokenIdx={node.tokenIdx!}
                    open={true}
                  />
                  <button
                    class="chip small-margin cg-chip-mono"
                    onClick={() => showDocument(node.docId!)}
                  >
                    <span>{node.docId}</span>
                    {node.pubYear && <span class="small-text"> {node.pubYear}</span>}
                  </button>
                </section>
              </Show>

              {/*  Neighbours  */}
              <Show when={neighbourEdges().length > 0}>
                {(_) => {
                  const edges = neighbourEdges();
                  const maxWeight = edges[0]?.edge.weight ?? 1;
                  return (
                    <section class="bottom-padding">
                      <details open={true}>
                        <summary><h4 class="bottom-padding">Neighbours ({edges.length})</h4></summary>
                        <For each={edges}>
                          {({ edge, neighbour }) => (
                            <div class="row max">
                              <div class="cg-nb-bar-wrap" style="width: 33%">
                                <div
                                  class="cg-nb-bar-fill primary"
                                  style={{ width: `${ (edge.weight / maxWeight) * 100 }%`, background: 'blue' }}
                                />
                              </div>
                              <span class="cg-nb-token"><q>{neighbour.label}</q></span>
                              <span class="cg-nb-score">{edge.weight.toFixed(3)}</span>
                            </div>
                          )}
                        </For>
                      </details>
                    </section>
                  );
                }}
              </Show>
            </article>
          );
        }}
      </Show>
    </aside >
  );
};

export default ConceptGraphSidebar;
