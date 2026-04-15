import { For } from "solid-js";
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
    return (
        <header class={"responsive surface-container-high " + styles.driftLegend} >
            <nav class="wrap small-padding">
                <For each={props.terms}>
                    {(term) => (
                        <label
                            style={
                                "background-color:" + props.colorScale(term) + ' !important; ' +
                                "color:" + props.colorScale(term) + ' !important; '
                            }
                            class={"chip checkbox small small-padding " + (
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
                        </label>
                    )}
                </For>
            </nav>
        </header >
    );
};
