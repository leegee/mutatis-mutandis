import { For } from "solid-js";
import { color } from "./DriftChart";
import { eeboStore } from "../stores/Eebo.store";
import styles from "./DriftChart.module.css";

type Props = {
    terms: string[];
    visible: Set<string>;
    onToggle: (term: string) => void;
};

export default function DriftLegend(props: Props) {
    return (
        <header class={"responsive surface-container border " + styles.driftLegend} >
            <nav class="wrap padding middle-align">
                <For each={props.terms}>
                    {(term) => (
                        <label
                            class={"chip checkbox small " + (
                                term === eeboStore.selected.token ? styles.selectedLegendTerm : ""
                            )}
                            onClick={(e) => {
                                e.preventDefault();
                                props.onToggle(term);
                            }}
                        >
                            <input
                                type="checkbox"
                                checked={props.visible.has(term)}
                                onChange={() => { }}
                            />
                            <span style={{ color: color(term) }}>
                                {term}
                            </span>
                        </label>
                    )}
                </For>
            </nav>
        </header >
    );
};
