import { Show } from "solid-js";
import styles from "./LineageGraph.module.css";
import type { TooltipState } from "./types";

type TooltipProps = {
    tooltip: TooltipState;
    concept?: string;
};

export default function Tooltip(props: TooltipProps) {
    return (
        <div
            class={styles.tooltip}
            style={{
                left: `${ props.tooltip.x + 14 }px`,
                top: `${ props.tooltip.y + 14 }px`,
            }}
        >
            <strong>
                {props.concept} · {props.tooltip.node.year}
            </strong>

            <div>
                cluster {props.tooltip.node.cluster} · mass {props.tooltip.node.size}
            </div>

            <Show when={props.tooltip.node.persistence_score !== undefined}>
                <div>
                    persistence{" "}
                    {props.tooltip.node.persistence_score!.toFixed(2)}

                    {props.tooltip.node.lineage_stable === false &&
                        " (drifted)"}
                </div>
            </Show>

            <Show when={props.tooltip.node.event_sample?.length}>
                <div class={styles.tooltipHint}>
                    click for events →
                </div>
            </Show>
        </div>
    );
}