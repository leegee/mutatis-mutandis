import { For, Show } from "solid-js";
import TextWindow from "../TextWindow";
import styles from "./LineageGraph.module.css";

import type { LineageNode } from "./types";

type DetailPanelProps = {
    node: LineageNode;
    concept?: string;
    onClose: () => void;
};

export default function DetailPanel(props: DetailPanelProps) {
    return (
        <aside class={styles.detailPanel}>
            <header class={styles.detailPanelHeader}>
                <div>
                    <strong>{props.concept}</strong>
                    <span>
                        {" "}
                        · {props.node.year} · cluster {props.node.cluster}
                    </span>
                </div>

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


            <Show
                when={props.node.event_sample?.length}
                fallback={
                    <p class={styles.detailPanelEmpty}>
                        No sampled events.
                    </p>
                }
            >
                <ul class={styles.eventList}>
                    <For each={props.node.event_sample}>
                        {ev => (
                            <li class={styles.eventItem}>
                                <TextWindow
                                    doc_id={String(ev.doc_id)}
                                    token_idx={String(ev.token_idx)}
                                />

                                <div class="tooltip top">
                                    {ev.doc_id} @ {ev.token_idx}
                                </div>

                                <Show when={ev.neighbours.length}>
                                    <ul class={styles.neighbourList}>
                                        <For each={ev.neighbours}>
                                            {nb => (
                                                <li class={styles.neighbourItem}>
                                                    <span class={styles.neighbourToken}>
                                                        {nb.token}
                                                    </span>

                                                    <span class={styles.neighbourMeta}>
                                                        {nb.doc_id} @ {nb.token_idx}
                                                        {" · "}
                                                        {nb.score.toFixed(2)}
                                                    </span>
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
