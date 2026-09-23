import type { LabelPoint, PointData } from "../components/ScatterPlot/types";
import { labelState, setLabelState } from "./labels.store";

export const labelsActions = {
    getLabels(concept: string) {
        return labelState.labels[concept] ?? [];
    },

    addLabel(concept: string, labelPoint: LabelPoint) {
        setLabelState("labels", concept, (prev = []) => [
            ...prev,
            labelPoint,
        ]);
    },

    getAcceptableCentroid(concept: string, points: PointData[]) {
        const centroid = computeCentroid(points);
        const labels = this.getLabels(concept);
        if (labels) {
            const minDist = labelState?.minCentroidDistance ?? 0.05;
            if (isTooClose(centroid, labels, minDist)) {
                console.log(`[label.actions] too close - bail out!`)
                return null;
            }
        }
        return centroid;
    },

    createFromCluster(concept: string, points: PointData[], cluster_label: string, description: string,) {
        const centroid = this.getAcceptableCentroid(concept, points);
        console.debug(`[label.actions] enter with points:`, JSON.stringify(points), "\nGot centroid", centroid)

        if (!centroid) throw new Error("No centroid for points")

        const labels = this.getLabels(concept);
        if (labels) {
            const minDist = labelState?.minCentroidDistance ?? 0.05;
            if (isTooClose(centroid, this.getLabels(concept), minDist)) {
                console.log(`[label.actions] too close - bail out!`)
                return false;
            }
        }

        const labelPoint: LabelPoint = {
            id: crypto.randomUUID(),
            cluster_label,
            description,
            nx: centroid.nx,
            ny: centroid.ny,
            gnx: centroid.gnx,
            gny: centroid.gny,
            type: "cluster_summary"
        };
        console.log(`[label.actions] label! ${ cluster_label }`, labelPoint)

        this.addLabel(concept, labelPoint);
        return true;
    }
};


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


// Trimmed-mean centroid hack
function computeCentroid(
    points: { gnx: number; gny: number; nx: number; ny: number }[],
    trimRatio = 0.1 // 10% trimmed from each side
) {
    console.log(`[computeCentroid]`);

    const nxArr: number[] = [];
    const nyArr: number[] = [];
    const gnxArr: number[] = [];
    const gnyArr: number[] = [];

    for (const p of points) {
        if (
            typeof p.nx === "undefined" ||
            typeof p.ny === "undefined" ||
            typeof p.gnx === "undefined" ||
            typeof p.gny === "undefined"
        ) {
            throw new Error("point nx/ny undefined");
        }

        nxArr.push(p.nx);
        nyArr.push(p.ny);
        gnxArr.push(p.gnx);
        gnyArr.push(p.gny);
    }

    const trimmedMean = (arr: number[]) => {
        const sorted = arr.slice().sort((a, b) => a - b);
        const n = sorted.length;

        const trim = Math.floor(n * trimRatio);
        const trimmed = sorted.slice(trim, n - trim);

        return trimmed.reduce((sum, v) => sum + v, 0) / trimmed.length;
    };

    const rv = {
        nx: trimmedMean(nxArr),
        ny: trimmedMean(nyArr),
        gnx: trimmedMean(gnxArr),
        gny: trimmedMean(gnyArr),
    };

    console.log(`[computeCentroid] rv`, rv);

    return rv;
}
