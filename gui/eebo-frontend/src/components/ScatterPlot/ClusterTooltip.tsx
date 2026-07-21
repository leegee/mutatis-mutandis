import type { LabelPoint, PointData } from "./types";

interface Props {
    point: PointData;
}

export default function ClusterTooltip(props: Props) {
    const point = () => props.point as any; // TODO LabelPoint?

    return (
        <>
            <header class="fill padding">
                <h2>{point().cluster_label ?? "No cluster"}</h2>
            </header>

            <div class="left-padding right-padding">
                {point().description && (
                    <div class="top-margin" style="width:100%">
                        {point().description}
                    </div>
                )}

                <div class={point().cluster_label
                    ? "top-margin bottom-margin small-text large-opacity"
                    : ""
                }>
                    <p>
                        Concept: {point().concept}
                        <br />
                        Cluster ID: {point().cluster_id}
                    </p>
                </div>
            </div>
        </>
    );
}
