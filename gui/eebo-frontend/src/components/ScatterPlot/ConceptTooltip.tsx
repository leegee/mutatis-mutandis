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
            </header>


            <div class="left-padding right-padding bottom-margin">
                <span class="large-opacity">
                    <span class="bold">
                        {point().pub_year}
                    </span>
                    <span class="large-padding">&mdash; </span>
                    {point().concept}
                </span>
            </div>

            <div class="left-padding right-padding">
                <span class="medium-opacity">
                    Document ID: {point().doc_id}
                    <br />
                    Token {point().token_idx}
                    <br />
                    Window: {point().window_id}
                </span>
            </div>

            <div class="left-padding right-padding">
                <span class="medium-opacity">
                    Cluster {point().cluster_id || "None"}
                </span>
            </div>

            <footer class="row padding fill bottom-padding top-padding" >
                <TextWindow
                    eventid={point().event_id}
                    style="font-size: 12pt; line-height: 1.6;"
                />
            </footer>
        </>
    );
}
