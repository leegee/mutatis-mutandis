import type { UmapPoint } from "./types";

interface Props {
    point: UmapPoint;
}

export default function ConceptTooltip(props: Props) {
    return (
        <div class="surface-container-high medium-elevate">
            <header class="bottom-margin fill">
                <h2 class="max padding">
                    {props.point.token ?? "No token"}
                </h2>
            </header>

            <div class="left-padding right-padding bottom-margin">
                <span class="large-opacity">
                    <span class="bold">
                        {props.point.pubYear ?? "Unknown year"}
                    </span>
                    <span class="large-padding">&mdash;</span>
                    Cluster {props.point.clusterId ?? "None"}
                </span>
            </div>
        </div>
    );
}
