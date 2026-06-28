import type { LabelPoint, PointData } from "../components/ScatterPlot/types";
import { labelState, setLabelState } from "./labels.store";

export const labelsActions = {
    createFromCluster(
        points: PointData[],
        text: string
    ) {
        console.log(`[label.actions] enter with points:`, JSON.stringify(points))
        const centroid = computeCentroid(points);
        console.debug(`[label.actions] centroid`, centroid)

        const labels = labelState.labels;
        console.debug(`[label.actions] labels`, labels)

        const minDist = labelState?.minCentroidDistance ?? 0.05;
        console.debug(`[label.actions] mid`, minDist)

        if (labels && isTooClose(centroid, labels, minDist)) {
            console.log(`[label.actions] too close - bail out!`)
            return false;
        }

        const labelPoint: LabelPoint = {
            id: crypto.randomUUID(),
            text,
            nx: centroid.nx,
            ny: centroid.ny,
            gnx: centroid.gnx,
            gny: centroid.gny,
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
    centroid: { nx: number; ny: number },
    labels: LabelPoint[],
    minDist: number
) {
    const minDist2 = minDist * minDist;

    return labels.some((l) => {
        const dx = l.nx - centroid.nx;
        const dy = l.ny - centroid.ny;
        return dx * dx + dy * dy < minDist2;
    });
}

function computeCentroid(points: { gnx: number; gny: number; nx: number; ny: number }[]) {
    console.log(`[computeCentroid]`)
    let nx = 0;
    let ny = 0;
    let gnx = 0;
    let gny = 0;

    for (const p of points) {
        console.log(`[computeCentroid] point ${ JSON.stringify(p) }`)
        if (typeof p.nx === "undefined" || typeof p.ny === "undefined"
            || typeof p.gnx === "undefined" || typeof p.gny === "undefined"
        ) {
            throw new Error("point nx/ny undefined");
        }
        nx += p.nx;
        ny += p.ny;
        gnx += p.gnx;
        gny += p.gny;
    }

    return {
        nx: nx / points.length,
        ny: ny / points.length,
        gnx: gnx / points.length,
        gny: gny / points.length,
    };
}


