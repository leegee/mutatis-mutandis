import { createMemo, createEffect, createSignal, onMount, onCleanup } from "solid-js";
import type { SliceView } from "../types";

export type SliceDensityFieldProps = {
    slice: SliceView;
    width?: number;
    height?: number;
};

type Vec2L = {
    x: number;
    y: number;
    label: string;
    weight: number;
    isTarget?: boolean;
};

const PAD = 32;

/* ---------------- PCA ---------------- */

function gaussian2D(dx: number, dy: number, h: number) {
    return Math.exp(-(dx * dx + dy * dy) / (2 * h * h));
}

function pca2D(points: number[][]) {
    const n = points.length;
    if (!n) return [];

    const dim = points[0].length;

    const mean = new Array(dim).fill(0);
    for (const p of points)
        for (let i = 0; i < dim; i++) mean[i] += p[i];

    for (let i = 0; i < dim; i++) mean[i] /= n;

    const cov = Array.from({ length: dim }, () => new Array(dim).fill(0));

    for (const p of points) {
        for (let i = 0; i < dim; i++) {
            for (let j = 0; j < dim; j++) {
                cov[i][j] += (p[i] - mean[i]) * (p[j] - mean[j]);
            }
        }
    }

    for (let i = 0; i < dim; i++)
        for (let j = 0; j < dim; j++)
            cov[i][j] /= n;

    const vars = cov.map((row, i) => ({ i, v: row[i] }))
        .sort((a, b) => b.v - a.v);

    const ax = vars[0].i;
    const ay = vars[1]?.i ?? ax;

    return points.map(p => ({
        x: p[ax],
        y: p[ay]
    }));
}

/* ---------------- layout ---------------- */

function normalizePoints(
    pts: { x: number; y: number }[],
    w: number,
    h: number
) {
    if (!pts.length) return [];

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const p of pts) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    }

    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;

    return pts.map(p => ({
        x: PAD + ((p.x - minX) / dx) * (w - PAD * 2),
        y: PAD + ((p.y - minY) / dy) * (h - PAD * 2)
    }));
}

/* ---------------- field ---------------- */

function computeField(points: Vec2L[], w: number, h: number, res = 3, hK = 40) {
    const wSteps = Math.floor(w / res);
    const hSteps = Math.floor(h / res);

    const field = Array.from({ length: hSteps }, () =>
        new Array(wSteps).fill(0)
    );

    let max = 0;

    for (let y = 0; y < hSteps; y++) {
        for (let x = 0; x < wSteps; x++) {

            const px = x * res;
            const py = y * res;

            let sum = 0;

            for (const p of points) {
                if (p.isTarget) continue;
                sum += p.weight * gaussian2D(px - p.x, py - p.y, hK);
            }

            field[y][x] = sum;
            if (sum > max) max = sum;
        }
    }

    return { field, max, res };
}

/* ---------------- render ---------------- */

function renderField(
    ctx: CanvasRenderingContext2D,
    field: number[][],
    max: number,
    res: number
) {
    const h = field.length;
    const w = field[0].length;

    const img = ctx.createImageData(w * res, h * res);

    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {

            const v = field[y][x] / (max || 1);
            const c = Math.pow(v, 0.65) * 255;

            for (let dy = 0; dy < res; dy++) {
                for (let dx = 0; dx < res; dx++) {

                    const i =
                        (y * res + dy) * w * res +
                        (x * res + dx);

                    img.data[i * 4 + 0] = c;
                    img.data[i * 4 + 1] = 40;
                    img.data[i * 4 + 2] = 140;
                    img.data[i * 4 + 3] = 255;
                }
            }
        }
    }

    ctx.putImageData(img, 0, 0);
}

function drawAxes(ctx: CanvasRenderingContext2D, w: number, h: number) {
    ctx.fillStyle = "rgba(255,255,255,0.8)";
    ctx.font = "10pt sans-serif";

    ctx.textAlign = "center";
    ctx.fillText("Semantic Proximity →", w / 2, h - 10);

    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Usage Intensity →", 12, 0);
    ctx.restore();
}

function selectTop(points: Vec2L[], k = 12) {
    return points
        .filter(p => !p.isTarget)
        .sort((a, b) => b.weight - a.weight)
        .slice(0, k);
}

function avoidCollisions(points: Vec2L[], d = 18) {
    const out: Vec2L[] = [];

    for (const p of points) {
        if (out.every(o => {
            const dx = p.x - o.x;
            const dy = p.y - o.y;
            return dx * dx + dy * dy > d * d;
        })) {
            out.push(p);
        }
    }

    return out;
}


export default function SliceDensityField(props: SliceDensityFieldProps) {
    let canvas!: HTMLCanvasElement;
    let container!: HTMLDivElement;

    const [size, setSize] = createSignal({ width: 0, height: 0 });

    onMount(() => {
        if (!container) return;

        const ro = new ResizeObserver(([entry]) => {
            setSize({
                width: entry.contentRect.width,
                height: entry.contentRect.height
            });
        });

        ro.observe(container);
        onCleanup(() => ro.disconnect());
    });

    const w = () => props.width ?? size().width;
    const h = () => props.height ?? size().height;

    const projected = createMemo<Vec2L[]>(() => {
        const n = props.slice.neighbors ?? [];

        const vectors = n.map(d => {
            const sim = d.similarity ?? 0;
            const count = Math.log1p(d.count ?? 1);
            return [sim, count, sim * count];
        });

        const raw = pca2D(vectors);
        const norm = normalizePoints(raw, w(), h());

        const neighbors = norm.map((p, i) => {
            const d = n[i];
            return {
                x: p.x,
                y: p.y,
                label: d.token,
                weight: Math.log1p(d.count ?? 1) * (d.similarity ?? 0.5)
            };
        });

        const target: Vec2L = {
            x: w() / 2,
            y: h() / 2,
            label: props.slice.token ?? "TARGET",
            weight: 2,
            isTarget: true
        };

        return [target, ...neighbors];
    });

    const draw = () => {
        if (!canvas) return;

        const width = w();
        const height = h();

        if (!width || !height) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const pts = projected();

        const { field, max, res } = computeField(pts, width, height);

        ctx.clearRect(0, 0, width, height);

        renderField(ctx, field, max, res);
        drawAxes(ctx, width, height);

        const labels = avoidCollisions(selectTop(pts, 15));

        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (const l of labels) {
            const tw = ctx.measureText(l.label).width;

            ctx.fillStyle = "#00D3";
            ctx.fillRect(l.x - tw / 2 - 2, l.y - 10, tw + 4, 18);

            ctx.fillStyle = "white";
            ctx.fillText(l.label, l.x, l.y);
        }

        const target = pts.find(p => p.isTarget);
        if (target) {
            const yearLabel =
                props.slice.slice_start === props.slice.slice_end
                    ? `${props.slice.slice_start ?? ""}`
                    : `${props.slice.slice_start}-${String(props.slice.slice_end).substring(2)}`;

            ctx.textAlign = "center";

            ctx.font = "bold 38pt sans-serif";
            ctx.fillStyle = "#001A";
            ctx.fillText(target.label, target.x, target.y - 20);

            ctx.font = "bold 32pt sans-serif";
            ctx.fillStyle = "#0029";
            ctx.fillText(yearLabel, target.x, target.y + 48);
        }
    };

    onMount(draw);

    createEffect(() => {
        projected();
        draw();
    });

    return (
        <div ref={container} style={{ width: "100%", height: "100%" }}>
            <canvas
                ref={canvas}
                width={w()}
                height={h()}
            />
        </div>
    );
}
