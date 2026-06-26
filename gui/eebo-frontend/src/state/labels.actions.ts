import type { LabelDataset, LabelPoint, PointData } from "../components/ScatterPlot/types";
import { labelState, setLabelState } from "./labels.store";

export const labelsActions = {
    createFromCluster(
        points: PointData[],
        text: string
    ) {
        console.log(`[label.actions] enter`, points)
        const centroid = computeCentroid(points);
        console.log(`[label.actions] centroid`, centroid)

        const labels = labelState.labels;
        console.log(`[label.actions] labels`, labels)

        const minDist = labelState?.minCentroidDistance ?? 0.05;
        console.log(`[label.actions] mid`, minDist)

        if (labels && isTooClose(centroid, labels, minDist)) {
            console.log(`[label.actions] too close - bail out!`)
            return false;
        }

        const labelPoint: LabelPoint = {
            id: crypto.randomUUID(),
            text,
            nx: centroid.x,
            ny: centroid.y,
            type: "cluster_summary"
        };
        console.log(`[label.actions] label! ${ text }`, labelPoint)

        addLabel(labelPoint);

        return true;
    }
};

function addLabel(labelPoint: LabelPoint) {
    console.log('Added label', labelPoint);
    setLabelState("labels", (prev) => [
        ...(prev ?? []),
        labelPoint,
    ]);
}

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
    console.log(`[computeCentroid]`)
    let x = 0;
    let y = 0;

    for (const p of points) {
        console.log(`[computeCentroid] point ${ p }`)
        x += p.nx;
        y += p.ny;
    }

    return {
        x: x / points.length,
        y: y / points.length,
    };
}


