import { createSignal, For, Match, Show, Switch } from "solid-js";
import type { ScaleOrdinal } from "d3-scale";

import { eeboStore } from "../stores/Eebo.store";
import styles from "./DriftChart.module.css";

type Props = {
    terms: string[];
    visible: Set<string>;
    onToggle: (term: string) => void;
    colorScale: ScaleOrdinal<string, string>;
};

export default function DriftLegend(props: Props) {
    const [iAmOpen, setIAmOpen] = createSignal(true);
    return (
        <nav class={"m l left " + styles.driftLegend + ' '
            + (iAmOpen() ? 'max surface-container-high' : ' throbbing transparent ' + styles.driftLegendMin)
        }>
            <header>
                <h6 class="row max">
                    <button
                        class={
                            "border extra circle transparent"
                            + (iAmOpen() ? '' : ' throbbing')
                        }
                        onclick={() => setIAmOpen(!iAmOpen())}
                    >
                        <Switch>
                            <Match when={iAmOpen()}>
                                <i>menu_open</i>
                            </Match>
                            <Match when={!iAmOpen()}>
                                <i>menu</i>
                            </Match>
                        </Switch>
                    </button>
                    <Show when={iAmOpen()}>
                        Filters
                    </Show>
                </h6>
            </header >

            <Show when={iAmOpen()}>
                <For each={props.terms}>
                    {(term) => (
                        <button
                            style={
                                "background-color:" + props.colorScale(term) + ' !important; ' +
                                "color:" + props.colorScale(term) + ' !important; '
                            }
                            class={"chip checkbox small-padding " + (
                                term === eeboStore.selected.token
                                    ? ('surface-container-highest large-elevate ' + String(styles.legendTerm) + ' ' + styles.selectedLegendTerm)
                                    : (' surface-container-high ' + String(styles.legendTerm))
                            )}
                            onClick={(e) => {
                                e.preventDefault();
                                props.onToggle(term);
                            }}
                        >
                            <input type="checkbox" checked={props.visible.has(term)} onChange={() => { }} />
                            <span> {term} </span>
                        </button>
                    )}
                </For>
            </Show>
        </nav >
    );
};
