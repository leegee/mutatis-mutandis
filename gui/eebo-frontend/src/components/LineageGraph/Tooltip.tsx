import { Show } from "solid-js";
import styles from "./LineageGraph.module.css";
import type { TooltipState } from "./types";

type TooltipProps = {
    tooltip: TooltipState;
    concept?: string;
};

export default function Tooltip(props: TooltipProps) {
    return (
        <aside
            class={styles.tooltip + " large-elevate surface-container-highest"}
            style={{
                left: `${ props.tooltip.x }px`,
                top: `${ props.tooltip.y }px`,
            }}
        >
            <strong>
                {props.concept} · {props.tooltip.node.year}
            </strong>

            <Show when={props.tooltip.node.context_profile?.length}>
                <div class={styles.tooltipSection}>
                    <div class={styles.tooltipHeading}>
                        context profile
                    </div>

                    <ul>
                        {props.tooltip.node.context_profile!.map(term => (
                            <li>
                                {term.token}
                                <span>
                                    {" "}
                                    {term.count}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            </Show>

            <Show when={props.tooltip.node.persistence_score !== undefined}>
                <div class={styles.tooltipMeta}>
                    persistence{" "}
                    {props.tooltip.node.persistence_score!.toFixed(2)}

                    {props.tooltip.node.lineage_stable === false &&
                        " (drifted)"}
                </div>
            </Show>

            <Show when={props.tooltip.node.event_sample?.length}>
                <div class={styles.tooltipHint}>
                    click for textual evidence →
                </div>
            </Show>
        </aside>
    );
}
