import { Show } from "solid-js";
import TextWindow from "../TextWindow";
import type { PointData } from "./types";

interface Props {
    point: PointData;
}

export default function ConceptTooltip(props: Props) {
    const point = () => props.point;

    return (
        <>
            <header class="bottom-margin fill">
                <h2 class="fill max">
                    <q>{point().token}</q>
                </h2>

                <Show when={point().depth}>
                    <div class="medium-opacity small-text no-space small small-margin tiny-padding">
                        <span class="max no-space small small-margin no-padding">
                            <span class="bold">{point().pub_year}</span>{" "}
                            {point().concept}
                            <sup class="medium-text">
                                {" "}{point().depth}
                            </sup>
                        </span>
                    </div>
                </Show>
            </header>

            <div class="left-padding right-padding">
                <span class="medium-opacity">
                    Doc: {point().doc_id} T {point().token_idx}
                    <br />
                    Win: {point().window_id} T {point().window_token_pos}
                </span>
            </div>

            <div class="left-padding right-padding">
                <span class="medium-opacity">
                    Cluster {point().cluster_label || "N/A"}
                </span>
            </div>

            <footer
                class="row padding fill"
                style="bottom-padding: 1em; top-padding: 1em"
            >
                <TextWindow
                    eventid={point().event_id}
                    style="font-size: 12pt; line-height: 1.6;"
                />
            </footer>
        </>
    );
}
