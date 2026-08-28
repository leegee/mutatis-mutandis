import { Show } from "solid-js";
import { computeTooltipStyle } from "../../lib/tooltipPosition";
import styles from "./Tooltip.module.css";
import type { TooltipState } from "./types";

type TooltipProps = {
    tooltip: TooltipState;
    concept?: string;
};

export default function Tooltip(props: TooltipProps) {
    return (
        <aside
            class={styles.tooltip + " large-elevate surface-container-highest padding"}
            style={computeTooltipStyle(props.tooltip.x, props.tooltip.y)}
        >
            <h6 class="bottom-margin">
                {props.concept} · {props.tooltip.node.year}
            </h6>

            <Show when={props.tooltip.node.retrieval_profile?.length}>
                <div class={styles.tooltipSection}>
                    <ul class={styles.tooltipList + " list no-space no-padding border"}>
                        {props.tooltip.node.retrieval_profile!.map((term) => (
                            <li>
                                <span>{term.token}</span>
                                <span>{term.count}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            </Show>

            <Show when={props.tooltip.node.persistence_score !== undefined}>
                <div class="padding">
                    persistence {props.tooltip.node.persistence_score!.toFixed(2)}
                    {props.tooltip.node.lineage_stable === false && " (drifted)"}
                </div>
            </Show>

            <Show when={props.tooltip.node.event_sample?.length}>
                <div class="medium-opacity info">click for textual evidence →</div>
            </Show>
        </aside>
    );
}
