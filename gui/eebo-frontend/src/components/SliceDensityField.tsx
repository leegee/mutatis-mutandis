import { type Component, createMemo, onCleanup, onMount } from "solid-js";
import type { SliceView } from "../types";
import { eeboStore, setEeboStore } from "../stores/Eebo.store";

type Neighbor = SliceView["neighbors"][number];

export type SliceDensityFieldProps = {
    slice: SliceView;
    width: number;
    height: number;
};

type Vec2 = { x: number; y: number };

function gaussian2D(dx: number, dy: number, h: number) {
    return Math.exp(-(dx * dx + dy * dy) / (2 * h * h));
}

/**
 * PCA (2D) via covariance eigenvectors
 * - deterministic
 * - slice-local geometry
 */
function pca2D(points: number[][]): Vec2[] {
    const n = points.length;
    if (n === 0) return [];

    const dim = points[0].length;

    // mean
    const mean = new Array(dim).fill(0);
    for (const p of points) {
        for (let i = 0; i < dim; i++) mean[i] += p[i];
    }
    for (let i = 0; i < dim; i++) mean[i] /= n;

    // covariance (diagonal approximation for stability + speed)
    const cov = new Array(dim).fill(0).map(() => new Array(dim).fill(0));

    for (const p of points) {
        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                cov[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]);
            }
        }
    }

    for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
            cov[i][j] /= n;
        }
    }

    // crude 2D projection: take two highest-variance axes
    const variances = cov.map((row, i) => ({ i, v: row[i] }));
    variances.sort((a, b) => b.v - a.v);

    const ax = variances[0].i;
    const ay = variances[1]?.i ?? variances[0].i;

    return points.map(p => ({
        x: p[ax],
        y: p[ay]
    }));
}

function computeField(
    points: Vec2[],
    width: number,
    height: number,
    resolution = 3,
    h = 35
) {
    const wSteps = Math.floor(width / resolution);
    const hSteps = Math.floor(height / resolution);

    const field: number[][] = Array.from({ length: hSteps }, () =>
        new Array(wSteps).fill(0)
    );

    let max = 0;

    for (let y = 0; y < hSteps; y++) {
        for (let x = 0; x < wSteps; x++) {

            const px = x * resolution;
            const py = y * resolution;

            let sum = 0;

            for (const p of points) {
                sum += gaussian2D(px - p.x, py - p.y, h);
            }

            field[y][x] = sum;
            if (sum > max) max = sum;
        }
    }

    return { field, max, resolution };
}

function renderField(
    ctx: CanvasRenderingContext2D,
    field: number[][],
    max: number,
    resolution: number
) {
    const h = field.length;
    const w = field[0].length;

    const img = ctx.createImageData(w * resolution, h * resolution);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {

            const v = field[y][x] / (max || 1);

            // perceptual ramp (better than linear blue/pink collapse)
            const c = Math.min(255, Math.floor(Math.pow(v, 0.65) * 255));

            for (let dy = 0; dy < resolution; dy++) {
                for (let dx = 0; dx < resolution; dx++) {

                    const ix =
                        (y * resolution + dy) * w * resolution +
                        (x * resolution + dx);

                    img.data[ix * 4 + 0] = c;      // R
                    img.data[ix * 4 + 1] = 40;     // G
                    img.data[ix * 4 + 2] = 140;    // B
                    img.data[ix * 4 + 3] = 255;    // A
                }
            }
        }
    }

    ctx.putImageData(img, 0, 0);
}

const SliceDensityField: Component<SliceDensityFieldProps> = (props) => {
    let canvasRef: HTMLCanvasElement | undefined;

    const neighbors = createMemo(() => props.slice.neighbors ?? []);

    const projected = createMemo(() => {
        const n = neighbors();

        // flatten embeddings from FAISS-like structure if available
        const vectors = n.map(d => {
            // fallback: simulate embedding if missing
            const sim = d.similarity ?? 0;
            const count = Math.log1p(d.count ?? 1);

            return [
                sim,
                count,
                sim * count
            ];
        });

        const pts2D = pca2D(vectors);

        return pts2D.map((p, i) => ({
            x: p.x * props.width,
            y: p.y * props.height
        }));
    });

    const draw = () => {
        if (!canvasRef) return;

        const ctx = canvasRef.getContext("2d");
        if (!ctx) return;

        const pts = projected();

        const { field, max, resolution } = computeField(
            pts,
            props.width,
            props.height,
            3,
            40
        );

        ctx.clearRect(0, 0, props.width, props.height);
        renderField(ctx, field, max, resolution);
    };

    onMount(() => {
        draw();
    });

    return (
        <article style={{ width: "100%", height: "100%", position: "relative" }}>
            <canvas
                ref={el => (canvasRef = el)}
                width={props.width}
                height={props.height}
                style={{ width: "100%", height: "100%" }}
            />
        </article>
    );
};

export default SliceDensityField;
