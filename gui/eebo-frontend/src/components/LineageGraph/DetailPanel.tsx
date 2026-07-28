import { For, Show } from "solid-js";
import TextWindow from "../TextWindow";
import styles from "./LineageGraph.module.css";

import type { LineageNode } from "./types";
import { showDocument } from "../../services/documentApi";

type DetailPanelProps = {
    node: LineageNode;
    concept?: string;
    onClose: () => void;
};

export default function DetailPanel(props: DetailPanelProps) {
    return (
        <aside class={styles.detailPanel + " left-padding right-padding"}>
            <header class={styles.detailPanelHeader + " middle-align"}>
                <h6 class="max medium-text">
                    <strong>{props.concept}</strong>
                    · {props.node.year}
                    <span class="max small-text">
                        {" "}
                        · cluster {props.node.cluster}
                    </span>
                </h6>

                <button
                    class={styles.detailPanelClose}
                    onClick={props.onClose}
                >
                    <i>close</i>
                </button>
            </header>


            <div class={styles.detailPanelMeta}>
                <span>
                    mass {props.node.size}
                </span>

                <Show when={props.node.persistence_score !== undefined}>
                    <span>
                        persistence{" "}
                        {props.node.persistence_score!.toFixed(2)}
                    </span>
                </Show>

                <Show when={props.node.lineage_stable === false}>
                    <span class={styles.driftedTag}>
                        drifted lineage
                    </span>
                </Show>

                <Show when={props.node.merged_from?.length}>
                    <span>
                        merged from lineage
                        {props.node.merged_from!.length > 1 ? "s" : ""}{" "}
                        {props.node.merged_from!.join(", ")}
                    </span>
                </Show>
            </div>


            <Show when={props.node.event_sample?.length} fallback={
                <p class={styles.detailPanelEmpty}> No sampled events. </p>
            } >
                <ul class={styles.eventList}>
                    <For each={props.node.event_sample}>
                        {ev => (
                            <li class={styles.eventItem}>
                                <div onDblClick={() => showDocument(ev.doc_id, ev.token_idx)}>
                                    <TextWindow
                                        doc_id={String(ev.doc_id)}
                                        token_idx={ev.token_idx}
                                    />
                                </div>

                                <div class="tooltip top max">
                                    {ev.doc_id} @ {ev.token_idx}<br />
                                    <small>Double-click the excerpt for the full text</small>
                                </div>

                                <Show when={ev.neighbours.length}>
                                    <ul class={"list no-space border"}>
                                        <For each={ev.neighbours}>
                                            {nb => (
                                                <li class={styles.neighbourItem}>
                                                    <div>
                                                        <span class={styles.neighbourToken}>
                                                            {nb.token}
                                                        </span>
                                                        <br />

                                                        <span class={styles.neighbourMeta}>
                                                            {"×"}
                                                            {nb.count}
                                                            {" · "}
                                                            {nb.max_score.toFixed(2)}
                                                        </span>
                                                    </div>

                                                    <ul class="list no-space border small-text">
                                                        <For each={nb.examples}>
                                                            {example => (
                                                                <li>
                                                                    {example.doc_id}
                                                                    {" @ "}
                                                                    {example.token_idx}
                                                                    {" · "}
                                                                    {example.score.toFixed(2)}
                                                                </li>
                                                            )}
                                                        </For>
                                                    </ul>
                                                </li>
                                            )}
                                        </For>
                                    </ul>
                                </Show>
                            </li>
                        )}
                    </For>
                </ul>
            </Show>
        </aside>
    );
}
