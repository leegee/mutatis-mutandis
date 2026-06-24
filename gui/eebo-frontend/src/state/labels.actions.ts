import type { LabelDataset, LabelPoint, PointData } from "../components/ScatterPlot/types";
import { labelState, setLabelState } from "./labels.store";

export const labelsActions = {
    setLabelDataset(dataset: LabelDataset) {
        setLabelState("labelDataset", dataset)
    },


    createFromCluster(
        points: PointData[],
        text: string
    ) {
        console.log(`[label.actions] enter`)
        const centroid = computeCentroid(points);
        console.log(`[label.actions] centroid`, centroid)
        const dataset = labelState.labelDataset;
        console.log(`[label.actions] dataset`, dataset)
        const minDist = dataset?.minCentroidDistance ?? 0.05;
        console.log(`[label.actions] mid`, minDist)

        if (dataset &&
            isTooClose(centroid, dataset.labels, minDist)
        ) {
            console.log(`[label.actions] too close - bail out!`)
            return false;
        }

        const label: LabelPoint = {
            id: crypto.randomUUID(),
            text,
            nx: centroid.x,
            ny: centroid.y,
        };
        console.log(`[label.actions] label! ${ label }`)

        setLabelState("labelDataset", "labels", (prev) => [
            ...(prev ?? []),
            label,
        ]);

        return true;
    }
};


function isTooClose(
    centroid: { x: number; y: number },
    labels: LabelPoint[],
    minDist: number
) {
    const minDist2 = minDist * minDist;

    return labels.some((l) => {
        const dx = l.nx - centroid.x;
        const dy = l.ny - centroid.y;
        return dx * dx + dy * dy < minDist2;
    });
}

function computeCentroid(points: { nx: number; ny: number }[]) {
    let x = 0;
    let y = 0;

    for (const p of points) {
        console.log(`[label.actions] point ${ p }`)
        x += p.nx;
        y += p.ny;
    }

    return {
        x: x / points.length,
        y: y / points.length,
    };
}


